import asyncio
import base64
import torch
import json
import os
import re
import time
from copy import deepcopy
from io import BytesIO
from multiprocessing import cpu_count
from typing import List, Tuple, Union

import numpy as np
from accelerate import Accelerator, DistributedType
from transformers import AutoTokenizer
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from openai import AsyncOpenAI, OpenAI
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

NUM_SECONDS_TO_SLEEP = 5

try:
    import vllm
except ImportError:
    vllm = None

TOOL_CALL_SYSTEM_PROMPT="""You are a helpful assistant.

# Tools

You may call the function tool shown below to assist with the user query.

You are provided with the function signature within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "resize_image", "name": "resize_image", "description": "Resize the image resolution.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `resize`: Double the resolution of the current image. You should only use this tool if you are unable to obtain the critical information needed to answer the question from the current resolution.", "enum": ["resize"], "type": "string"}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>
For each function call, return a json object with the function name and the corresponding argument within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

TOOL_CALL_MULTI_TRUN_PROMPT="Please carefully analyze the content returned from the image resize tool in combination with the original question and image from the user, continue your reasoning process inside <think> and </think> and then write your final answer inside <answer> and </answer>."

ERROR_INFO_MULTI_TURN_PROMPT="Please analyze the error information obtained from the function tool and adjust your response. Countinue your reasoning process inside <think> and </think>."

def resize_image(image: Image.Image, save_path=None):
    original_width, original_height = image.size
    # Calculate the new dimensions (double the size)
    new_width = original_width * 2
    new_height = original_height * 2
    print(f"[TOOL CALL RESIZE IMAGE]: NEW_IMAGE_SIZE: {(new_width, new_height)}.")
    # Resize the image
    resized_image = image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
    if save_path:
        # Save the enlarged image
        resized_image.save(save_path)
    return resized_image

def prepare_tool_call_inputs(json_objects: list):
    for obj in json_objects:
        action_type = obj['arguments']['action']
        assert action_type in ["resize"], f"Unknown Tool Type: {action_type}. Available function tools are: `resize`"
        assert len(json_objects) == 1, f"You should only call function `resize` once per function call."
    return action_type

@register_model("vllm_tool")
class VLLM_TOOL(lmms):
    def __init__(
        self,
        model_version: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.7,
        batch_size: int = 1,
        timeout: int = 60,
        max_images: int = 32,
        prompt: str = 'tool_call',
        enable_tool_call: bool = False,
        max_generation_round: bool = 2,
        downsample_image: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # Manually set a image token for GPT4V so that we can search for it
        # and split the text and image
        # Here we just use the same token as llava for convenient
        self.model_version = model_version
        self.max_images = max_images
        self.enable_tool_call = enable_tool_call
        self.max_generation_round = max_generation_round
        self.downsample_image = downsample_image
        accelerator = Accelerator()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_version)
        self.inference_engine = vllm.LLM(
            model=self.model_version,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=32768,
            enable_prefix_caching=True,
            max_model_len=32768,
            limit_mm_per_prompt={"image": max_images}
        )
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device
        self.batch_size_per_gpu = int(batch_size)
        self.prompt = prompt

    def make_conversation(self, problem, system_prompt, prompt_template):
        problem = prompt_template.format(Question=problem)
        prompt = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": problem,
            },
        ]
        return prompt

    def extract_responses_list(
        self, 
        tokenizer, 
        input_ids: torch.Tensor,
        multi_turn_response_mask: torch.Tensor # 0,0,0,...,1,1,1,...,0,0,0,...,1,1,1
    ) -> list:
        # Tensor Method
        diff = torch.diff(multi_turn_response_mask, prepend=torch.tensor([0], device=multi_turn_response_mask.device))
        starts = torch.where(diff == 1)[0]
        mask_appended = torch.cat([multi_turn_response_mask, torch.tensor([0], device=multi_turn_response_mask.device)], dim=0)
        diff_end = torch.diff(mask_appended)
        ends = torch.where(diff_end == -1)[0] - 1
        segments = []
        for s, e in zip(starts, ends):
            segments.append(input_ids[s:e+1].tolist())

        # Decode each segment
        # decoded_responses = [tokenizer.decode(seg, skip_special_tokens=True) for seg in segments]
        decoded_responses = tokenizer.batch_decode(segments, skip_special_tokens=True)
        return decoded_responses

    # Function to encode the image
    def encode_image(self, image: Union[Image.Image, str], downsample_image=False):
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy()

        # Ensure minimum dimensions of 28x28
        width, height = img.size
        if downsample_image:
            new_width = width // 2
            new_height = height // 2
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        width, height = img.size
        if width < 28 or height < 28:
            scale = max(28 / width, 28 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return img

    # Function to encode the video
    def encode_video(self, video_path, max_frames_num=8):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)

        # Ensure the last frame is included
        if total_frame_num - 1 not in uniform_sampled_frames:
            uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

        return base64_frames

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]

        if self.prompt == 'tool_call':
            system_prompt = TOOL_CALL_SYSTEM_PROMPT
            prompt_template = "Answer the question based on the image provided. You must conduct reasoning within <think> and </think> first in each of your reasoning steps. You may call ONE function tool per step to help you better solve the problem. Place the function tool within <tool_call> and </tool_call> at the end of each step to perform a function call. You should continue your reasoning process based on the content returned by the function tool. Once you confirm your final answer, place the final answer inside <answer> and </answer>. For mathematical or multiple-choice problem, wrap the answer value or choice with \\boxed{{}}. Here is the image and question:\n{Question}"
        elif self.prompt == 'raw_prompt':
            system_prompt = "You are a helpful assistant."
            prompt_template = "Answer the question based on the image provided. You must conduct reasoning within <think> and </think> first in each of your reasoning steps. You may call ONE function tool per step to help you better solve the problem. Place the function tool within <tool_call> and </tool_call> at the end of each step to perform a function call. You should continue your reasoning process based on the content returned by the function tool. Once you confirm your final answer, place the final answer inside <answer> and </answer>. For mathematical or multiple-choice problem, wrap the answer value or choice with \\boxed{{}}. Here is the image and question:\n{Question}"
        else:
            print(f"Invalid prompt type: {self.prompt}")
            raise NotImplementedError

        for batch_requests in batched_requests:
            batched_vllm_inputs = []
            multi_turn_response_mask = []
            prefix_prompt_lengths = []
            for idx in range(len(batch_requests)):
                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = batch_requests[idx].arguments
                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 16384
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0
                if "top_p" not in gen_kwargs or gen_kwargs["top_p"] <= 0 or gen_kwargs["top_p"] >= 1:
                    gen_kwargs["top_p"] = 0.95

                params = {
                    "temperature": gen_kwargs["temperature"],
                    "max_tokens": 8192, #gen_kwargs["max_new_tokens"],
                    "top_p": gen_kwargs["top_p"],
                    "n": 1
                }
                sampling_params = vllm.SamplingParams(**params)

                visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                if None in visuals:
                    visuals = []
                    imgs = []
                else:
                    visuals = self.flatten(visuals)
                    imgs = []  # multiple images or frames for video
                    orginal_imgs = []
                    for visual in visuals:
                        if isinstance(visual, str) and (".mp4" in visual or ".avi" in visual or ".mov" in visual or ".flv" in visual or ".wmv" in visual):
                            frames = self.encode_video(visual)
                            imgs.extend(frames)
                        elif isinstance(visual, str) and (".jpg" in visual or ".jpeg" in visual or ".png" in visual or ".gif" in visual or ".bmp" in visual or ".tiff" in visual or ".webp" in visual):
                            img = self.encode_image(visual)
                            imgs.append(img)
                        elif isinstance(visual, Image.Image):
                            img = self.encode_image(visual, self.downsample_image)
                            original_img = self.encode_image(visual)
                            imgs.append(img)
                            orginal_imgs.append(original_img)
                if "<image>" not in contexts:
                    assert isinstance(contexts, str), "Contexts type should be str!"
                    contexts = "<image>\n" + contexts
                chat = self.make_conversation(contexts, system_prompt, prompt_template)
                prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
                raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
                raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
                # vllm_input = {'prompt_token_ids': deepcopy(raw_prompt_ids), 'multi_modal_data': {"image": deepcopy(imgs)}}
                vllm_input = {'prompt_token_ids': deepcopy(raw_prompt_ids), 'multi_modal_data': {"image": deepcopy(imgs)}, 'original_image': orginal_imgs}
                prefix_length = len(raw_prompt_ids)
                batched_vllm_inputs.append(vllm_input)
                multi_turn_response_mask.append([torch.zeros(prefix_length)]) # [torch.Tensor(prefix_length,)]
                prefix_prompt_lengths.append(prefix_length)

            to_generate = list(range(len(batched_vllm_inputs)))
            max_image_num = self.max_images
            current_iteration = 0
            while current_iteration < self.max_generation_round and len(to_generate) > 0:
                idx_to_gen = [] # list of vllm_inputs, at first the length is B'*R
                for i in to_generate:
                    idx_to_gen.append(batched_vllm_inputs[i])
                eval_logger.info(f"[Round #{current_iteration} Rollout START] For THIS round, We hava {len(idx_to_gen)} trajs to complete ...")
                outputs = self.inference_engine.generate(
                    prompts=idx_to_gen,  # list of dict
                    sampling_params=sampling_params,
                    use_tqdm=False
                )
                response = [] # list of tuple, B'*R, valid(no-pad) response_ids with unequal length
                for output in outputs:
                    for sample_id in range(len(output.outputs)):
                        # HACK: filter > (voc_size + specidal_token_num) token_ids, 151664 for qwen model
                        _token_ids = output.outputs[sample_id].token_ids
                        filtered_token_ids = [token_id for token_id in _token_ids if token_id <= 151664]    # NOTE: <tool_call>: 151657, </tool_call>: 151658
                        if 151645 not in filtered_token_ids:
                            # replace the last token with <|im_end|> if no <|im_end|> in response,
                            # this is to ensure successful execution of get_final_eos_mask in multi-turn scenario
                            filtered_token_ids[-1] = 151645
                        response.append(filtered_token_ids)

                # attach model responses to vllm_inputs
                assert len(to_generate) == len(response)

                idx_to_remove = []
                id_tool_query_mapping = {}
                for i_gen, response_ in zip(to_generate, response):
                    # update conversation
                    response_ = list(response_)
                    batched_vllm_inputs[i_gen]['prompt_token_ids'] += response_
                    multi_turn_response_mask[i_gen].append(torch.ones(len(response_)))
                    # [TOOL CALL TRIGGER] We check model's last turn response, if not any tool called, then remove this traj from to_generate
                    decoded_resp_ = self.tokenizer.decode(response_, skip_special_tokens=True)
                    pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
                    tool_call_contents = pattern.findall(decoded_resp_)
                    if len(tool_call_contents) > 0:
                        if len(batched_vllm_inputs[i_gen]['multi_modal_data']['image']) >= max_image_num:   # If the current traj has already reached max_image_num, but still try to call tool, we should remove this traj.
                            idx_to_remove.append(i_gen)
                            eval_logger.info(f"Traj {i} exceeds maximum function tool call num {len(batched_vllm_inputs[i]['multi_modal_data']['image'])}")
                        assert str(i_gen) not in id_tool_query_mapping.keys()
                        error_info = None
                        try:
                            json_pattern = re.compile(r'\{.*?\}\}')
                            json_objects = []
                            for content in tool_call_contents:
                                json_strings = json_pattern.findall(content)
                                json_objects.extend([json.loads(json_str) for json_str in json_strings])
                            tool_type = prepare_tool_call_inputs(json_objects)
                        except Exception as e:
                            eval_logger.info(str(e))
                            error_info = str(e)
                            tool_type = None
                        id_tool_query_mapping[str(i_gen)] = {
                            "tool_type": tool_type,
                            "error_info": error_info,
                        }
                    # Direct Answer
                    else:
                        # remove this traj from to_generate
                        idx_to_remove.append(i_gen)
                        # NOTE: to_generate.remove(i_gen) # DO NOT .remove() in for loop
                    # eval_logger.info(f"[Round #{current_iteration}] i_gen: {i_gen} | resp: {self.tokenizer.decode(response_, skip_special_tokens=True)}")
                if to_generate and id_tool_query_mapping:   # Make sure to PRINT when to_generate and id_tool_query_mapping is not None
                    eval_logger.info(f"[Round #{current_iteration}] Example Generation: to_generate[0]: {to_generate[0]} | response[0]: {self.tokenizer.decode(response[0], skip_special_tokens=True)}")
                    eval_logger.info(f"[Round #{current_iteration} Rollout Tool Call Trigger] For THIS round, ids {next(iter(id_tool_query_mapping))} need to apply function tool using: {id_tool_query_mapping[next(iter(id_tool_query_mapping))]} ...")
                else:
                    eval_logger.info(f"[Round #{current_iteration} Rollout Tool Call Trigger] No ids need to apply function tool for this round.")
                # update 'to_generate'
                for x in idx_to_remove:
                    to_generate.remove(x)

                eval_logger.info(f"[Round #{current_iteration} Rollout END] For NEXT round, We hava {len(to_generate)} trajs to complete ...")

                # [Call Function Tool]
                function_tool_results = []
                for i_todo in to_generate:
                    assert str(i_todo) in id_tool_query_mapping.keys()
                    # image_to_resize = batched_vllm_inputs[i_todo]['multi_modal_data']['image'][-1]
                    image_to_resize = batched_vllm_inputs[i_todo]['original_image']
                    tool_type = id_tool_query_mapping[str(i_todo)]['tool_type']
                    error_info = id_tool_query_mapping[str(i_todo)]["error_info"]
                    if tool_type == "resize":
                        # tool_outputs = resize_image(image_to_resize)   # Here we can add a save_path to visualize the annotated images
                        tool_outputs = image_to_resize
                    else:
                        tool_outputs = error_info
                        eval_logger.info(f"Tool type {tool_type} is not implemented.")
                    function_tool_results.append(tool_outputs)

                # [Process Tool Call Results]
                to_generate_ = to_generate.copy() # make a copy since we will be modifying to_generate
                assert len(to_generate_) == len(function_tool_results)

                for i_gen_, tool_call_result_ in zip(to_generate_, function_tool_results):
                    if isinstance(tool_call_result_, list):
                        # Construct Next Round Prompt
                        # FIXME: Only support image resize tool
                        tool_call_prompt_message = "<|im_start|>user\n" + "<tool_response>\nThe resized image is shown below:\n"
                        for i in range(len(tool_call_result_)):
                            if isinstance(tool_call_result_[i], Image.Image):
                                tool_call_prompt_message += "<|vision_start|><|image_pad|><|vision_end|>\n"
                        tool_call_prompt_message += "</tool_response>\n" + TOOL_CALL_MULTI_TRUN_PROMPT + "<|im_end|>\n<|im_start|>assistant\n"
                        next_turn_prompt_ids = self.tokenizer.encode(tool_call_prompt_message)
                        # update conversation
                        batched_vllm_inputs[i_gen_]['prompt_token_ids'] += next_turn_prompt_ids # this might go over response length, but we will cut it later by 'max_total_response_length'
                        batched_vllm_inputs[i_gen_]['multi_modal_data']['image'].extend(tool_call_result_)
                        multi_turn_response_mask[i_gen_].append(torch.zeros(len(next_turn_prompt_ids)))
                    else:
                        tool_call_prompt_message = "<|im_start|>user\n" + tool_call_result_ + ERROR_INFO_MULTI_TURN_PROMPT + "<|im_end|>\n<|im_start|>assistant\n"
                        next_turn_prompt_ids = self.tokenizer.encode(tool_call_prompt_message)
                        batched_vllm_inputs[i_gen_]['prompt_token_ids'] += next_turn_prompt_ids # this might go over response length, but we will cut it later by 'max_total_response_length'
                        multi_turn_response_mask[i_gen_].append(torch.zeros(len(next_turn_prompt_ids), dtype=torch.int64))
                # update iteration count
                current_iteration += 1
            
            assert self.enable_tool_call
            response_text = []
            for i_ in range(len(batched_vllm_inputs)):
                first_round_prompt_length = prefix_prompt_lengths[i_]
                generation_response_ids = batched_vllm_inputs[i_]['prompt_token_ids'][first_round_prompt_length:]
                generation_response_masks = torch.cat(multi_turn_response_mask[i_][1:], dim=0)
                valid_indices = generation_response_masks.nonzero(as_tuple=True)[0]
                valid_generation_response_ids = [generation_response_ids[i] for i in valid_indices.tolist()]
                generation_text = self.tokenizer.decode(valid_generation_response_ids, skip_special_tokens=True)
                response_text.append(generation_text)

            assert len(response_text) == len(batch_requests)
            res.extend(response_text)
            pbar.update(len(batch_requests))

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "GPT4V not support"

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")