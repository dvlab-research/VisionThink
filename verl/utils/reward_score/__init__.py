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
# from verl.utils.reward_score import gsm8k, math, prime_math, prime_code


def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    reward_fn_key = "val_general_qa_reward_fn" if (extra_info.get("val_general_qa_reward_fn", None) is not None and kwargs.get("mode", None) == "val") else "general_qa_reward_fn"
    if data_source in [
            'numina_aops_forum', 'numina_synthetic_math', 'numina_amc_aime', 'numina_synthetic_amc', 'numina_cn_k12',
            'numina_olympiads'
    ]:
        from verl.utils.reward_score import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["open_source_no_require_high", "open_source_require_high", "open_source", "open_source_rulebase"]:
        if extra_info[reward_fn_key] == 'general_qa_gpt':
            from verl.utils.reward_score import general_qa_gpt
            res = general_qa_gpt.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
        elif extra_info[reward_fn_key] == 'general_qa_tool':
            from verl.utils.reward_score import general_qa_tool
            res = general_qa_tool.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
        elif extra_info['general_qa_reward_fn'] == 'rule_based_tool':
            from . import rule_based_tool
            res = rule_based_tool.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
        elif extra_info[reward_fn_key] == 'general_qa_verifier':
            from verl.utils.reward_score import general_qa_verifier
            res = general_qa_verifier.compute_score(kwargs['accuracy_score'], solution_str, ground_truth, extra_info)
    else:
        raise NotImplementedError

    # if isinstance(res, (int, float, bool)):
    #     return float(res)
    # else:
    #     return float(res[0])
    return res
