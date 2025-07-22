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

class NaiveMultiThreadsRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

        self.extra_info = kwargs.get("extra_info", {})
        self.mode = kwargs.get("mode", None)
        self.gpt_threads = kwargs.get("gpt_threads", 100)

    def process_single(self, args):

        index, data_item = args
        time_start = time.time()

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

        # print("extra_info: ", extra_info)
        question = data_item.non_tensor_batch['raw_prompt']
        result = self.compute_score(
            prompt=question,
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            mode=self.mode,
        )

        if isinstance(result, dict):
            return (index, result)
        else:
            score, acc_score, format_score = result

            time_end = time.time()

            # print("time consumed: {}".format(time_end - time_start))

            # return (score, acc_score, format_score, valid_response_length, data_source, prompt_str, response_str, ground_truth, time_end - time_start)
            return (index, score, acc_score, format_score, valid_response_length, data_source, prompt_str, response_str, ground_truth, time_end - time_start)

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

        n_threads = self.gpt_threads

        print("gpt_threads: ", n_threads)

        executor = ThreadPoolExecutor(n_threads)
        futures = []
        for i in range(len(data)):
            futures.append(executor.submit(self.process_single, (i, data[i])))
        results = [f.result() for f in as_completed(futures)]

        results = sorted(results, key=lambda x: x[0])
        results = [result[1:] for result in results]

        # results = [self.process_single(data[i]) for i in range(len(data))]

        time_consume_list = []
        invalid_uids = []
        # for i, (score, acc_score, format_score, valid_response_length, data_source, prompt_str, response_str, ground_truth, time_consume) in enumerate(results):
        for i, result in enumerate(results):

            if isinstance(result[0], dict) and result[0]['is_filter'] == True:
                invalid_uids.append(data[i].non_tensor_batch['uid'])
                continue
            
            (score, acc_score, format_score, valid_response_length, data_source, prompt_str, response_str, ground_truth, time_consume) = result
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
            
            time_consume_list.append(time_consume)

        time_consume_list = sorted(time_consume_list, reverse=True)
        print("top 10 time consuming in reward fn: ", time_consume_list[:10])

        print(f"there are {len(invalid_uids)} invalid samples in this batch: {invalid_uids[:5]}")

        time_end = time.time()

        print("total time: ", time_end - time_start)

        return reward_tensor, acc_reward_tensor, format_reward_tensor, invalid_uids

