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
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from verl.workers.reward_manager.utils import reward_func_timeout_ray
import ray
from ray.exceptions import GetTimeoutError  

class NaiveMultiThreadsV2RewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

        self.extra_info = kwargs.get("extra_info", {})
        self.mode = kwargs.get("mode", None)
        self.timeout_seconds = self.extra_info.get("timeout_seconds", 10)

    def process_single(self, question, data_source, response_str, ground_truth, extra_info):

        time_start = time.time()

        result = self.compute_score(
            prompt=question,
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            mode=self.mode,
        )

        time_end = time.time()

        if isinstance(result, dict):
            return result
        
        score, acc_score, format_score = result
        return score, acc_score, format_score, time_end - time_start


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

        futures = []
        question_list = []
        data_source_list = []
        prompt_str_list = []
        response_str_list = []
        ground_truth_list = []
        extra_info_list = []
        valid_response_length_list = []
        uid_list = []
        for i in range(len(data)):

            data_item = data[i]
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

            # import pdb; pdb.set_trace()
            uid = data_item.non_tensor_batch['uid']

            # print("extra_info: ", extra_info)
            question = data_item.non_tensor_batch['raw_prompt'] if 'raw_prompt' in data_item.non_tensor_batch else None

            future = reward_func_timeout_ray.remote(
                self.process_single, self.timeout_seconds, question, data_source, response_str, ground_truth, extra_info
            )
            futures.append(future)
            question_list.append(question)
            data_source_list.append(data_source)
            prompt_str_list.append(prompt_str)
            response_str_list.append(response_str)
            ground_truth_list.append(ground_truth)
            extra_info_list.append(extra_info)
            valid_response_length_list.append(valid_response_length)
            uid_list.append(uid)

        invalid_uids = []
        time_consume_list = []
        
        for i, future in enumerate(futures):
            try:
                task_result = ray.get(future, timeout=self.timeout_seconds)
                result = task_result
                
                if isinstance(result, dict) and result['is_filter'] == True:
                    invalid_uids.append(uid_list[i])
                    continue

                score, acc_score, format_score, time_consume = result
                time_consume_list.append(time_consume)

            except GetTimeoutError:
                print(f"Timeout processing item {i} (gold='{str(ground_truth_list[i])[:50]}...', prediction='{str(response_str_list[i])[:50]}...'). Using default score.")
                invalid_uids.append(uid_list[i])
                time_consume_list.append(self.timeout_seconds)
                continue
            except Exception as e:
                print(f"Error processing item {i} (gold='{str(ground_truth_list[i])[:50]}...', prediction='{str(response_str_list[i])[:50]}...'): {e}")
                import traceback
                traceback.print_exc()
                ray.cancel(future, force=True)
                invalid_uids.append(uid_list[i])
                time_consume_list.append(0)
                continue

            valid_response_length = valid_response_length_list[i]
            prompt_str = prompt_str_list[i]
            response_str = response_str_list[i]
            ground_truth = ground_truth_list[i]
            
            reward_tensor[i, valid_response_length - 1] = score
            acc_reward_tensor[i, valid_response_length - 1] = acc_score
            format_reward_tensor[i, valid_response_length - 1] = format_score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < 5:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", (score, acc_score, format_score))
                # import pdb; pdb.set_trace()
            
            time_consume_list.append(time_consume)

        time_consume_list = sorted(time_consume_list, reverse=True)
        print("top 10 time consuming in reward fn: ", time_consume_list[:10])

        print(f"there are {len(invalid_uids)} invalid samples in this batch: {invalid_uids[:5]}")

        time_end = time.time()

        print("total time: ", time_end - time_start)

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