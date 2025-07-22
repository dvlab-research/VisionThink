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

from omegaconf import ListConfig
import os
from typing import List, Union, Optional
import copy
import pandas as pd
from collections import defaultdict
import yaml
import json
import math
import random


import io
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
import datasets

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

prompt_template_dict = {
    "original": "{Question}",
    "vanilla": "{Question}\nOutput the thinking process within <think> </think> tags and final answer within <answer> </answer> tags. The final answer should contain \\boxed{{}}.",
    "tool_agent": "Answer the question based on the image provided. You must conduct reasoning within <think> and </think> first in each of your reasoning steps. You may call ONE function tool per step to help you better solve the problem. Place the function tool within <tool_call> and </tool_call> at the end of each step to perform a function call. You should continue your reasoning process based on the content returned by the function tool. Once you confirm your final answer, place the final answer inside <answer> and </answer>. For mathematical or multiple-choice problem, wrap the answer value or choice with \\boxed{{}}. Here is the image and question:\n{Question}",
}

def make_conversation_multimodal(sources, system_prompt=None, prompt_type="vanilla"):
    for source in sources:
        problem = prompt_template_dict[prompt_type].format(Question=source["problem"])
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
        source["prompt"] = prompt

    return sources

def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        if key == 'raw_prompt_ids': # FIXME: hack to cope with dataset whose input_ids is all the same
            array_list = np.empty(len(val), dtype=object)
            for i, lst in enumerate(val):
                array_list[i] = lst
            non_tensors[key] = array_list
        else:
            non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}



def process_image(image: dict, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512, tool_call=False, use_tgt_size=False, tgt_w=None, tgt_h=None, use_tool=False):
    # from io import BytesIO
    from PIL import Image
    
    if tgt_w and tgt_h:     
        image = image.resize((tgt_w, tgt_h), resample=Image.Resampling.LANCZOS)
        if use_tool and not use_tgt_size:      # Double the resolution of the image if the base model cannot handle this problem based on the current resolution
            tgt_w = tgt_w * 2
            tgt_h = tgt_h * 2
            image = image.resize((tgt_w, tgt_h), resample=Image.Resampling.LANCZOS)
        assert image.width >= 28 and image.height >= 28, "Qwen image size should be larger than 28 * 28."
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.LANCZOS)

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.LANCZOS)

    if tool_call:
        downsample_factor = 2
        width, height = image.width // downsample_factor, image.height // downsample_factor
        image = image.resize((width, height), resample=Image.Resampling.LANCZOS)

    assert image.width >= 28 and image.height >= 28, "Qwen image size should be larger than 28 * 28."

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


class MultiModalDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 data_path: str,
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key='prompt',
                 image_key='images',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 system_prompt: str = None,
                 max_pixels: int = 2048 * 2048,
                 min_pixels: int = 512 * 512,
                 mask_blank: bool = False,
                 use_3drope: bool = True,
                 prompt_type: str = "vanilla",
                 general_qa_reward_fn: str = 'general_qa_gpt',
                 tool_call: bool = False,
                 use_tgt_size: bool = False,
                 use_raw_image: bool = False,
                ):

        self.data_path = copy.deepcopy(data_path)
        self.original_parquet_files = copy.deepcopy(data_path)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation

        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        
        self.mask_blank = mask_blank
        self.use_3drope = use_3drope
        self.prompt_type = prompt_type
        self.general_qa_reward_fn = general_qa_reward_fn
        self.tool_call = tool_call
        self.use_tgt_size = use_tgt_size
        self.use_raw_image = use_raw_image

        self.serialize_dataset = False
        self._read_files_and_tokenize()



    def _read_files_and_tokenize(self):

        list_data_dict = datasets.load_from_disk(self.data_path)

        print(f">>> Original Data Size: {len(list_data_dict)}")

        tokenizer = self.tokenizer
        prompt_key = self.prompt_key

        def add_prompt(example):
            problem = prompt_template_dict[self.prompt_type].format(Question=example["problem"])
            example["prompt"] = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": problem}
            ]
            return example

        list_data_dict = list_data_dict.map(add_prompt, num_proc=256)

        def is_valid_length(example):
            return len(tokenizer.apply_chat_template(example[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length

        self.list_data_dict = list_data_dict.filter(is_valid_length, num_proc=256)

        print(f'>>> Filtered Dataset Size: {len(self.list_data_dict)}')

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_data_path') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = copy.deepcopy(self.list_data_dict[item])

        chat = row_dict.pop(self.prompt_key)


        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        if self.image_key in row_dict:  # expand image token
            images = row_dict[self.image_key]
            tgt_w = tgt_h = use_tool = None
            if 'info' in row_dict:
                info = row_dict['info']
                if 'tgt_width' in info and 'tgt_height' in info and 'use_tool' in info:
                    tgt_w, tgt_h, use_tool = info['tgt_width'], info['tgt_height'], info['use_tool']
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            try:
                row_dict['multi_modal_data'] = {'image': [process_image(row_dict[self.image_key], self.max_pixels, self.min_pixels, self.tool_call, self.use_tgt_size, tgt_w, tgt_h, use_tool)]}
            except Exception as e:
                print(str(e))
                return self.__getitem__(item+1) if item + 1 < len(self) else self.__getitem__(0)
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                              self.processor.image_token)
        else:
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        if self.mask_blank and images[0].split("/")[-1].split(".")[0] == "blank_image":
            image_token_id = self.tokenizer.encode(self.processor.image_token)[0]
            attention_mask[input_ids == image_token_id] = 0

        if self.use_raw_image:
            if tgt_w is None and tgt_h is None:
                assert len(row_dict['multi_modal_data']['image']) == 1, f"Before appending the orginal image into multi_modal_data, len(row_dict['multi_modal_data']['image']) should be 1, but got {len(row_dict['multi_modal_data']['image'])}."
                tgt_w, tgt_h = row_dict['multi_modal_data']['image'][0].size
            try:
                row_dict['multi_modal_data']['image'].extend([process_image(row_dict[self.image_key], self.max_pixels, self.min_pixels, self.tool_call, self.use_tgt_size, tgt_w * 2, tgt_h * 2, use_tool)])
            except Exception as e:
                print(str(e))
                return self.__getitem__(item+1) if item + 1 < len(self) else self.__getitem__(0)
                
        if self.use_3drope and self.image_key in row_dict:
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)
            # Add raw image at the end of multimodal_dataset to avoid <|image_pad|> error
            row_dict.pop(self.image_key)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        if self.tool_call and use_tool is not None:
            row_dict['use_tool'] = use_tool

        ### Add for CustomRewardManager ###


        if self.general_qa_reward_fn in ["general_qa_gpt", "general_qa_tool", "general_qa_verifier"]:

            row_dict['ground_truth'] = row_dict.pop('original_answer')
            for key in ['solution', 'reformat_answers', 'response']:
                if key in row_dict:
                    row_dict.pop(key)
        else:
            row_dict['ground_truth'] = row_dict.pop('solution')

        # encode prompts without chat template
        if self.return_raw_chat:
            assert chat[-1]['role'] == 'user'
            row_dict['raw_prompt'] = chat[-1]['content']

        # add index for each prompt
        index = row_dict.get("doc_id", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'list_data_dict' in state:
                del state['list_data_dict']
            return state
        return self.__dict__.copy()


if __name__ == '__main__':
    from verl.utils.fs import copy_to_local
    from verl.utils import hf_tokenizer, hf_processor
    local_path = copy_to_local("Qwen/Qwen2.5-VL-7B-Instruct")
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none
    SYSTEM_PROMPT="You FIRST think about the reasoning process step by step as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in <answer> </answer> tags."
    dataset = MultiModalDataset(
        data_path="generalqa.yaml",
        tokenizer=tokenizer,
        processor=processor,
        prompt_key='prompt',
        image_key='images',
        max_prompt_length=4096,
        filter_prompts=True,
        return_raw_chat=False,
        truncation='error',
        system_prompt=SYSTEM_PROMPT,
        max_pixels=2048 * 2048,
        min_pixels=512 * 512,
        use_3drope=True,
        tool_call=True,
        use_tgt_size=False,
        use_raw_image=True,
    )
    example = dataset[0]
    print("example.keys:", example.keys())
    print(len(dataset))