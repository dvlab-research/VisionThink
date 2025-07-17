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


import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
import re
from mathruler.grader import extract_boxed_content, grade_answer

### Easy-R1 Math Reward ###
def math_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0


def math_acc_reward(prompt_str: str, predict_str: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict_str)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def math_compute_score(prompt_str: str, predict_str: str, ground_truth: str) -> float:
    return 0.9 * math_acc_reward(prompt_str, predict_str, ground_truth) + 0.1 * math_format_reward(predict_str)

### R1-V Reward ###
def r1v_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0

def r1v_accuracy_reward(prompt_str: str, predict_str: str, ground_truth: str) -> float:
    try:
        ground_truth = ground_truth.strip()
        content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
        given_answer = content_match.group(1).strip() if content_match else predict_str.strip()
        if grade_answer(given_answer, ground_truth):
            return 1.0
    except Exception:
        pass
    return 0.0

def r1v_compute_score(prompt_str: str, predict_str: str, ground_truth: str) -> float:
    return 0.9 * r1v_accuracy_reward(prompt_str, predict_str, ground_truth) + 0.1 * r1v_format_reward(predict_str)

### MMRL Reward ###
def mmrl_accuracy_reward(prompt_str: str, predict_str: str, ground_truth: str) -> float:
    from verl.utils.reward_score.gpt import match_by_gpt4o
    try:
        question_match = re.search(r'user\n(.*?)\nassistant', prompt_str, re.DOTALL)
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if question_match and answer_match:
            question_str = question_match.group(1).strip()
            answer_str = answer_match.group(1).strip()
            score = match_by_gpt4o(question_str, answer_str, ground_truth)
            if score == None:
                return 0.0
            return score
        else:
            return 0.0
    except Exception:
        print("GPT4o Evaluation Failed, set accuracy reward to 0.")
        pass
    return 0.0

def mmrl_compute_score(prompt_str: str, predict_str: str, ground_truth: str) -> float:
    return 0.9 * mmrl_accuracy_reward(prompt_str, predict_str, ground_truth) + 0.1 * r1v_format_reward(predict_str)

class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, num_examine: int, compute_score: str, mode: str, num_threads: int = 64):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score_type = compute_score
        if mode == "train":
            if compute_score == "math":
                self.compute_score = math_compute_score
            elif compute_score == "r1v":
                self.compute_score = r1v_compute_score
            elif compute_score == "mmrl":
                from concurrent.futures import ThreadPoolExecutor
                self.compute_score = mmrl_compute_score
                self.executor = ThreadPoolExecutor(num_threads)
            elif compute_score == "mmrl_acc":
                from concurrent.futures import ThreadPoolExecutor
                self.compute_score = mmrl_accuracy_reward
                self.executor = ThreadPoolExecutor(num_threads)
            elif compute_score == "mm_eureka":
                self.compute_score = math_compute_score
            else:
                raise NotImplementedError()
        elif mode == "val": # Only evaluate accuracy scores when using "val" mode
            if compute_score == "math":
                self.compute_score = math_acc_reward
            elif compute_score == "r1v":
                self.compute_score = r1v_accuracy_reward
            elif compute_score == "mmrl":
                from concurrent.futures import ThreadPoolExecutor
                self.compute_score = mmrl_accuracy_reward
                self.executor = ThreadPoolExecutor(num_threads)
            elif compute_score == "mmrl_acc":
                from concurrent.futures import ThreadPoolExecutor
                self.compute_score = mmrl_accuracy_reward
                self.executor = ThreadPoolExecutor(num_threads)
            elif compute_score == "mm_eureka":
                self.compute_score = math_acc_reward
            else:
                raise NotImplementedError()

    def __call__(self, data: DataProto) -> torch.Tensor:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print = 0

        if self.compute_score_type in ["mmrl", "mmrl_acc",]:
            futures = []    # gpt4o evaluation
            prompt_str_list = []
            response_str_list = []
            ground_truth_list = []
            valid_response_length_list = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = data_item.non_tensor_batch["ground_truth"]

            if self.compute_score_type in ["math", "r1v", "mm_eureka"]:
                score = self.compute_score(prompt_str, response_str, ground_truth)
                reward_tensor[i, valid_response_length - 1] = score
                if already_print < self.num_examine:
                    already_print += 1
                    print("[prompt]", prompt_str)
                    print("[response]", response_str)
                    print("[ground_truth]", ground_truth)
                    print("[score]", score)

            elif self.compute_score_type in ["mmrl", "mmrl_acc",]:
                prompt_str_list.append(prompt_str)
                response_str_list.append(response_str)
                ground_truth_list.append(ground_truth)
                valid_response_length_list.append(valid_response_length)
                futures.append(self.executor.submit(self.compute_score, prompt_str, response_str, ground_truth))
        
        if self.compute_score_type in ["mmrl", "mmrl_acc",]:
            scores = torch.tensor([f.result() for f in futures])
            valid_response_lengths = torch.stack(valid_response_length_list) - 1    # -1 to act as idx
            reward_tensor.scatter_(1, valid_response_lengths.unsqueeze(1), scores.unsqueeze(1))
            while already_print < self.num_examine:
                print("[prompt]", prompt_str_list[already_print])
                print("[response]", response_str_list[already_print])
                print("[ground_truth]", ground_truth_list[already_print])
                print("[score]", scores[already_print].item())
                already_print += 1

        return reward_tensor