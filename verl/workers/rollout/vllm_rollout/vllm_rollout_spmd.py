# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import re
import json
import numpy as np
from PIL import Image
from typing import List
from copy import deepcopy
from tqdm import tqdm, trange
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from typing import Any, Union
from verl import DataProto
from verl.workers.rollout.base import BaseRollout
from .function_tools import prepare_tool_call_inputs, resize_image
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version
from verl.models.transformers.qwen2_vl import get_rope_index
from verl.trainer.constants import TOOL_CALL_MULTI_TRUN_PROMPT, ERROR_INFO_MULTI_TURN_PROMPT
from verl.utils.torch_functional import get_eos_mask, get_final_eos_mask, pad_2d_list_to_length, pad_sequence_to_length

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics

# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def pad_to_max_stack(tensor_list: List[torch.Tensor], pad_token_id: int, dim: int) -> torch.Tensor:
    assert all([t.ndim == 1 for t in tensor_list])
    max_len = max([t.size(0) for t in tensor_list])
    padded_tensor_list = []
    for t in tensor_list:
        padded_tensor_list.append(torch.cat([t,torch.tensor([pad_token_id] * (max_len-t.size(0)), device=t.device, dtype=t.dtype)],dim=0))
    return torch.stack(padded_tensor_list, dim=dim)


class vLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, processor, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            processor: the processor to process multi_modal_inputs
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            seed=0,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)
        self.pad_token_id = tokenizer.pad_token_id
        # add tokenizer
        self.tokenizer = tokenizer
        # add processor
        self.processor = processor
        self.merge_length = self.processor.image_processor.merge_size ** 2

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        if 'multi_modal_data' in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'),
                                                        non_tensor_batch.pop('multi_modal_data')):
                vllm_inputs.append({'prompt_token_ids': raw_prompt_ids, 'multi_modal_data': multi_modal_data})
        else:
            vllm_inputs = [{
                'prompt_token_ids': raw_prompt_ids
            } for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids')]

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False)

        # TODO(sgm): disable logprob when recompute_log_prob is enable
        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

        response = []
        for output in outputs:
            for sample_id in range(len(output.outputs)):
                response.append(output.outputs[sample_id].token_ids)

        response = pad_2d_list_to_length(response, self.pad_token_id,
                                         max_length=self.config.response_length).to(idx.device)

        if self.config.n > 1 and do_sample:
            idx = _repeat_interleave(idx, self.config.n)
            attention_mask = _repeat_interleave(attention_mask, self.config.n)
            position_ids = _repeat_interleave(position_ids, self.config.n)
            batch_size = batch_size * self.config.n
        # NOTE: We repeat 'multi_modal_data'
        rollout_num = self.config.n if do_sample else 1     # validation: n=1; train: n=self.config.n
        if 'multi_modal_data' in vllm_inputs[0]:
            repeated_multi_modal_data = [vllm_input['multi_modal_data'] for vllm_input in vllm_inputs for _ in range(rollout_num)]
            non_tensor_batch['multi_modal_data'] = np.array(repeated_multi_modal_data)

        seq = torch.cat([idx, response], dim=-1)

        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        response_length = response.size(1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids_list = []
            for prompt_with_response, attn_mask, multi_modal_data in zip(seq, attention_mask, repeated_multi_modal_data):
                image_inputs = self.processor.image_processor(multi_modal_data['image'], return_tensors='pt')
                image_grid_thw = image_inputs['image_grid_thw']
                pos_ids = get_rope_index(
                    self.processor,
                    input_ids=prompt_with_response,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attn_mask,
                )
                position_ids_list.append(pos_ids)
            position_ids = torch.stack(position_ids_list, dim=0)
        else:
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
            delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
            # TODO(sgm): fix position_ids on right_pad
            # prompt: left pad + response: right pad
            # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
            # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
            response_position_ids = position_ids[:, -1:] + delta_position_id
            position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

class vLLMRollout_MultiTurn_ToolCall(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, processor, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.
        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            processor: the processor to process multi_modal_inputs
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=config.max_total_response_length,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            limit_mm_per_prompt=self.config.get('limit_mm_per_prompt', {'image': 1}),
            seed=0,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)
        self.pad_token_id = tokenizer.pad_token_id
        self.max_generation_round = self.config.get('max_generation_round', 1)
        self.use_raw_image = self.config.use_raw_image
        # add tokenizer
        self.tokenizer = tokenizer
        # add processor
        self.processor = processor
        self.merge_length = self.processor.image_processor.merge_size ** 2

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:

        print(f">>> vLLM Rollout Starts ...")

        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()
        
        idx = prompts.batch['input_ids']  # (bs*tp, max_prompt_length), left padding with |end_of_text|
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']  # (bs*tp, max_prompt_length), left padding 0
        position_ids = prompts.batch['position_ids']  # (bs*tp, max_prompt_length), left padding 0

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']  # [151645, 151643] -> ｜im_end｜, |end_of_text|

        batch_size = idx.size(0)  # B'
        input_prompt_generation_mask = torch.zeros_like(idx, dtype=attention_mask.dtype, device=attention_mask.device) # (B'*R, max_prompt_length), all 0

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        
        n = 1 if prompts.meta_info.get('validate', False) else self.config.n  # TODO: for validate, do_sample=False
        ##### Initialization #####
        vllm_inputs = [] # B*R, list of dict, into -> vllm.engine, each dict with keys: 'prompt_token_ids', 'multi_modal_data', the values are 'raw_prompt_ids' and [PIL.Image]
        multi_turn_response_mask = [] # B*R, list of list of Tensor, for distinguish 'USER tokens' & 'ASSISTANT tokens'
        prefix_prompt_lengths = [] # B*R, list of int, record first round prompt of all trajs
        if self.use_raw_image:
            resized_image_inputs = []

        # We manually repeart trajs for rollout, since some trajs need multi-round self.inference_engine.generate() with `sampling_n=1`
        if 'multi_modal_data' in non_tensor_batch:
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'), non_tensor_batch.pop('multi_modal_data')):
                prefix_length = len(raw_prompt_ids)
                if self.use_raw_image:
                    assert len(multi_modal_data['image']) == 2, f"multi_modal_data should contain the downscaled image and a upscaled resized image, but got len(multi_modal_data['image']): {len(multi_modal_data['image'])}."
                    resized_image = multi_modal_data['image'].pop(-1)
                for _ in range(n):
                    # NOTE: use deepcopy to seperate variables
                    if self.use_raw_image:
                        resized_image_inputs.append(resized_image)
                    vllm_inputs.append(
                        {'prompt_token_ids': deepcopy(raw_prompt_ids), 'multi_modal_data': deepcopy(multi_modal_data)} # raw_prompt_ids: list
                    )
                    multi_turn_response_mask.append(
                        [torch.zeros(prefix_length, dtype=attention_mask.dtype, device=attention_mask.device)], # USER, Mark as 0
                    ) # [torch.Tensor(prefix_length,)]
                    prefix_prompt_lengths.append(
                        prefix_length
                    )
        ##### Loop Setting #####
        to_generate = list(range(batch_size * n))  # B*R, all trajs' index
        worker_trajs_count = len(to_generate)
        max_image_num = self.config.get('limit_mm_per_prompt', {'image': 1})['image']
        max_iterations = self.max_generation_round
        # Add pbar for better monitoring
        with tqdm(total=worker_trajs_count, desc="Worker Rollout Progress", unit="task") as pbar:
            current_iteration = 0
            while current_iteration < max_iterations and len(to_generate) > 0: 
                # Prepare prompts to generation
                idx_to_gen = [] # list of vllm_inputs, at first the length is B'*R
                for i in to_generate:
                    idx_to_gen.append(vllm_inputs[i])
                print(f"[Round #{current_iteration} Rollout START] For THIS round, We hava {len(idx_to_gen)} trajs to complete ...")
                # users can customize different sampling_params at different run
                with self.update_sampling_params(n=1):  # TODO: for validate, do_sample=False
                    outputs = self.inference_engine.generate(
                        prompts=idx_to_gen,  # list of dict
                        sampling_params=self.sampling_params,
                        use_tqdm=False
                    )
                response = [] # list of tuple, B'*R, valid(no-pad) response_ids with unequal length
                for output in outputs:
                    for sample_id in range(len(output.outputs)):
                        # HACK: filter > (voc_size+specidal_token_num) token_ids, 151664 for qwen model
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
                    vllm_inputs[i_gen]['prompt_token_ids'] += response_
                    multi_turn_response_mask[i_gen].append(torch.ones(len(response_), dtype=attention_mask.dtype, device=attention_mask.device)) # ASSISTANT, Mark as 1
                    # [TOOL CALL TRIGGER] We check model's last turn response, if not any tool called, then remove this traj from to_generate
                    decoded_resp_ = self.tokenizer.decode(response_, skip_special_tokens=True)
                    pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
                    tool_call_contents = pattern.findall(decoded_resp_)
                    if len(tool_call_contents) > 0:
                        if len(vllm_inputs[i_gen]['multi_modal_data']['image']) >= max_image_num:   # If the current traj has already reached max_image_num, but still try to call tool, we should remove this traj after this generation round.
                            idx_to_remove.append(i_gen)
                            print(f"Traj {i_gen} exceeds maximum function tool call num {len(vllm_inputs[i_gen]['multi_modal_data']['image'])}")
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
                            print(str(e))
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
                    # print(f"[Round #{current_iteration}] i_gen: {i_gen} | resp: {self.tokenizer.decode(response_, skip_special_tokens=True)}")
                if to_generate and id_tool_query_mapping:   # Make sure to PRINT when to_generate and id_tool_query_mapping is not None
                    print(f"[Round #{current_iteration}] Example Generation: to_generate[0]: {to_generate[0]} | response[0]: {self.tokenizer.decode(response[0], skip_special_tokens=True)}")
                    print(f"[Round #{current_iteration} Rollout Tool Call Trigger] For THIS round, ids {next(iter(id_tool_query_mapping))} need to apply function tool using: {id_tool_query_mapping[next(iter(id_tool_query_mapping))]} ...")
                else:
                    print(f"[Round #{current_iteration} Rollout Tool Call Trigger] No ids need to apply function tool for this round.")
                # update 'to_generate'
                for x in idx_to_remove:
                    to_generate.remove(x)

                print(f"[Round #{current_iteration} Rollout END] For NEXT round, We hava {len(to_generate)} trajs to complete ...")

                # [Call Function Tool]
                function_tool_results = []
                for i_todo in to_generate:
                    assert str(i_todo) in id_tool_query_mapping.keys()
                    image_to_resize = vllm_inputs[i_todo]['multi_modal_data']['image'][-1]
                    tool_type = id_tool_query_mapping[str(i_todo)]['tool_type']
                    error_info = id_tool_query_mapping[str(i_todo)]["error_info"]
                    if error_info is not None:
                        tool_outputs = f"<tool_response>\nERROR occurs during the function tool call. Error Information: {error_info}.\n</tool_response>\n"
                    else:
                        if self.use_raw_image:
                            tool_outputs = resized_image_inputs[i_todo]
                        else:
                            tool_outputs = resize_image(image_to_resize)   # Here we can add a save_path to visualize the annotated images
                    function_tool_results.append(tool_outputs)
            
                # [Process Tool Call Results]
                to_generate_ = to_generate.copy() # make a copy since we will be modifying to_generate
                assert len(to_generate_) == len(function_tool_results)

                for i_gen_, tool_call_result_ in zip(to_generate_, function_tool_results):
                    if isinstance(tool_call_result_, Image.Image):
                        # Construct Next Round Prompt
                        # FIXME: Only support image resize tool
                        tool_call_prompt_message = "<|im_start|>user\n" + "<tool_response>\nThe resized image is shown below:\n<|vision_start|><|image_pad|><|vision_end|>\n</tool_response>\n" + TOOL_CALL_MULTI_TRUN_PROMPT + "<|im_end|>\n<|im_start|>assistant\n"
                        next_turn_prompt_ids = self.tokenizer.encode(tool_call_prompt_message)
                        # update conversation
                        vllm_inputs[i_gen_]['prompt_token_ids'] += next_turn_prompt_ids # this might go over response length, but we will cut it later by 'max_total_response_length'
                        vllm_inputs[i_gen_]['multi_modal_data']['image'].append(tool_call_result_)
                        multi_turn_response_mask[i_gen_].append(torch.zeros(len(next_turn_prompt_ids), dtype=attention_mask.dtype, device=attention_mask.device)) # USER, Mark as 0
                    else:
                        tool_call_prompt_message = "<|im_start|>user\n" + tool_call_result_ + ERROR_INFO_MULTI_TURN_PROMPT + "<|im_end|>\n<|im_start|>assistant\n"
                        next_turn_prompt_ids = self.tokenizer.encode(tool_call_prompt_message)
                        vllm_inputs[i_gen_]['prompt_token_ids'] += next_turn_prompt_ids # this might go over response length, but we will cut it later by 'max_total_response_length'
                        multi_turn_response_mask[i_gen_].append(torch.zeros(len(next_turn_prompt_ids), dtype=attention_mask.dtype, device=attention_mask.device)) # USER, Mark as 0
                # update pbar
                pbar.update(worker_trajs_count - len(to_generate))
                # update iteration count
                current_iteration += 1

        # re-build response
        response = [] # B'*R, torch.Tensors with unequal lengths
        response_generation_mask = [] # B'*R, torch.Tensors with unequal lengths but align with 'response'
        for i_ in range(batch_size * n):
            # for each traj, we skip first-round prompt_ids/attention_mask
            first_round_prompt_length = prefix_prompt_lengths[i_]
            # Repeat <|image_pad|> token id for modeling_qwen2vl
            generation_response_ids = vllm_inputs[i_]['prompt_token_ids'][first_round_prompt_length:]
            resized_image_inputs = self.processor.image_processor(vllm_inputs[i_]['multi_modal_data']['image'][1:], return_tensors='pt')    # NOTE: The fisrt image is the original image, here we only take the resized image into account
            image_grid_thws = resized_image_inputs['image_grid_thw']
            all_response_masks = torch.cat(multi_turn_response_mask[i_][1:], dim=0).tolist()
            index, image_pad_token, magic_num = 0, 151655, 654321
            while image_pad_token in generation_response_ids:
                image_pad_token_pos = generation_response_ids.index(image_pad_token)
                image_pad_token_repeat_num = image_grid_thws[index].prod() // self.merge_length
                # update response_tensor_ids
                generation_response_ids[image_pad_token_pos : image_pad_token_pos + 1] = [magic_num] * image_pad_token_repeat_num
                # update all_response_masks
                all_response_masks[image_pad_token_pos : image_pad_token_pos + 1] = [0] * image_pad_token_repeat_num
                index += 1
            generation_response_ids = [image_pad_token if x == magic_num else x for x in generation_response_ids]
            all_response = torch.tensor(generation_response_ids, device=idx.device, dtype=idx.dtype)
            all_response_masks = torch.tensor(all_response_masks, dtype=torch.int64, device=attention_mask.device)
            response.append(all_response)
            response_generation_mask.append(all_response_masks) # at least we have single-turn conversation
            assert response[i_].shape[0] == response_generation_mask[i_].shape[0], f"Shape mismatched between resp_id and resp_mask! response[i_].shape[0]: {response[i_].shape[0]}, response_generation_mask[i_].shape[0]: {response_generation_mask[i_].shape[0]}"
        assert len(response) == len(response_generation_mask), "Length mismatched between response and response_generation_mask!"

        # attention_mask:       prompt           response
        #                 [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        response = pad_to_max_stack(response, self.pad_token_id, dim=0) # Tensor, (B'*R, padded_length), padded_length is the max length of samples in list
        response_generation_mask = pad_to_max_stack(response_generation_mask, 0, dim=0) # Tensor, (B'*R, padded_length)
        assert all([response.size(dim) == response_generation_mask.size(dim) for dim in range(response.ndim)])

        # cut or pad to max length
        # all should be (B*R, self.config.max_total_response_length)
        if response.shape[1] > self.config.max_total_response_length:
            response = response[:,:self.config.max_total_response_length]
            response_generation_mask = response_generation_mask[:,:self.config.max_total_response_length]
        elif response.shape[1] < self.config.max_total_response_length:
            response = pad_sequence_to_length(response, self.config.max_total_response_length, self.pad_token_id)
            response_generation_mask = pad_sequence_to_length(response_generation_mask, self.config.max_total_response_length, 0)

        # All for 1st USER prompt
        if self.config.n > 1 and do_sample:
            idx = _repeat_interleave(idx, self.config.n) # (B, max_prompt_length) -> (B*R, max_prompt_length)
            attention_mask = _repeat_interleave(attention_mask, self.config.n)
            position_ids = _repeat_interleave(position_ids, self.config.n)
            batch_size = batch_size * self.config.n
            # we also need to repeat 'input_prompt_generation_mask'
            input_prompt_generation_mask = _repeat_interleave(input_prompt_generation_mask, self.config.n) # (B, max_prompt_length) -> (B*R, max_prompt_length), all 0

        # NOTE: We repeat 'multi_modal_data'
        if 'multi_modal_data' in vllm_inputs[0]:
            repeated_multi_modal_data = [vllm_input['multi_modal_data'] for vllm_input in vllm_inputs] 
            non_tensor_batch['multi_modal_data'] = np.array(repeated_multi_modal_data)

        del vllm_inputs

        seq = torch.cat([idx, response], dim=-1) # (B*R, max_prompt_length + max_total_response_length)

        # FIXME: function get_final_eos_mask cannot handle cases that when there is no <|im_end|> in the given response
        # response_attention_mask = get_final_eos_mask(response_id=response, eos_token=[151645], dtype=attention_mask.dtype) # HACK: for qwen, <|im_end|> is 151645
        # attention_mask: (...,0,0,0,1,1,1), response_attention_mask: (1,1,1,0,0,0,...)
        response_attention_mask = get_final_eos_mask(response_id=response, eos_token=[151645], dtype=attention_mask.dtype) # HACK: for qwen, |im_end| is 151645
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        multi_turn_response_mask = torch.cat([input_prompt_generation_mask, response_generation_mask], dim=-1)

        response_length = response.size(1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids_list = []
            for prompt_with_response, attn_mask, multi_modal_data in zip(seq, attention_mask, repeated_multi_modal_data):
                image_inputs = self.processor.image_processor(multi_modal_data['image'], return_tensors='pt')
                image_grid_thw = image_inputs['image_grid_thw']
                pos_ids = get_rope_index(
                    self.processor,
                    input_ids=prompt_with_response,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attn_mask,
                )
                position_ids_list.append(pos_ids)
            position_ids = torch.stack(position_ids_list, dim=0)
        else:
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
            delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
            # TODO(sgm): fix position_ids on right_pad
            # prompt: left pad + response: right pad
            # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
            # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
            response_position_ids = position_ids[:, -1:] + delta_position_id
            position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        # NOTE: .contiguous() for broadcast
        batch = TensorDict(
            {
                'prompts': idx.contiguous(),
                'responses': response.contiguous(),
                'input_ids': seq.contiguous(),  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask.contiguous(),
                'position_ids': position_ids.contiguous(),
                'multi_turn_response_mask': multi_turn_response_mask.contiguous()
            },
            batch_size=batch_size
        )

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        print(f">>> vLLM Rollout Ends ...")

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)