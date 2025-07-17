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

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
# from verl.utils.reward_score.openr1 import format_reward, acc_reward
import torch
import time


class NaiveRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

        self.extra_info = kwargs.get("extra_info", {})
        print("gpt_extract_answer: ", self.extra_info.get("gpt_extract_answer", "Not set"))

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        acc_reward_tensor = reward_tensor.clone()
        format_reward_tensor = reward_tensor.clone()

        already_print_data_sources = {}

        time_start = time.time()

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            extra_info = self.extra_info.update(extra_info) if extra_info else self.extra_info

            question = data_item.non_tensor_batch.get('raw_prompt', None)
            # print("extra_info: ", extra_info)

            score, acc_score, format_score, invalid_uids = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                prompt=question,
            )
            reward_tensor[i, valid_response_length - 1] = score
            acc_reward_tensor[i, valid_response_length - 1] = acc_score
            format_reward_tensor[i, valid_response_length - 1] = format_score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)

        time_end = time.time()
        print("reward time: {}".format(time_end - time_start))

        return reward_tensor, acc_reward_tensor, format_reward_tensor, invalid_uids

    # def cal_format_reward_for_logging(self, data: DataProto):
    #     format_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

    #     for i in range(len(data)):
    #         data_item = data[i]  # DataProtoItem

    #         prompt_ids = data_item.batch['prompts']

    #         prompt_length = prompt_ids.shape[-1]

    #         valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
    #         valid_prompt_ids = prompt_ids[-valid_prompt_length:]

    #         response_ids = data_item.batch['responses']
    #         valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
    #         valid_response_ids = response_ids[:valid_response_length]

    #         # decode
    #         prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
    #         response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
    #         format_score = format_reward(
    #             response_str, self.extra_info
    #         )
    #         format_reward_tensor[i, valid_response_length - 1] = format_score
    #     return format_reward_tensor

    # def cal_acc_reward_for_logging(self, data: DataProto):
    #     acc_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
    #     for i in range(len(data)):
    #         data_item = data[i]  # DataProtoItem

    #         prompt_ids = data_item.batch['prompts']

    #         prompt_length = prompt_ids.shape[-1]

    #         valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
    #         valid_prompt_ids = prompt_ids[-valid_prompt_length:]

    #         response_ids = data_item.batch['responses']
    #         valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
    #         valid_response_ids = response_ids[:valid_response_length]

    #         # decode
    #         prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
    #         response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

    #         ground_truth = data_item.non_tensor_batch['ground_truth']

    #         acc_score = acc_reward(
    #             response_str, ground_truth, self.extra_info
    #         )
    #         acc_reward_tensor[i, valid_response_length - 1] = acc_score
    #     return acc_reward_tensor