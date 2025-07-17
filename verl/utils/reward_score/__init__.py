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
    if data_source == 'openai/gsm8k':
        from verl.utils.reward_score import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH']:
        from verl.utils.reward_score import math
        res = math.compute_score(solution_str, ground_truth)
    elif data_source in [
            'numina_aops_forum', 'numina_synthetic_math', 'numina_amc_aime', 'numina_synthetic_amc', 'numina_cn_k12',
            'numina_olympiads'
    ]:
        from verl.utils.reward_score import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ['codecontests', 'apps', 'codeforces', 'taco']:
        from verl.utils.reward_score import prime_code
        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ['hiyouga/geometry3k']:
        from verl.utils.reward_score import geo3k
        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in ['SynthLabsAI/Big-Math-RL-Verified', 'DigitalLearningGmbH/MATH-lighteval', 'HuggingFaceH4/aime_2024', 'AI-MO/aimo-validation-amc', 'HuggingFaceH4/MATH-500', "mathvision_testmini", "generated_manual_count_max_5_squares", "generated_manual_count_max_10_squares", "generated_manual_count_max_15_squares", "generated_manual_count_max_20_squares", "manual_count_val_mathv", "charxiv", "chartqa", "ttv13_rule_base_use_tool", "ttv13_rule_base_direct_answer"]:
        if extra_info[reward_fn_key] == 'rule_based_tool':
            from verl.utils.reward_score import rule_based_tool
            res = rule_based_tool.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
        else:
            from verl.utils.reward_score import openr1
            res = openr1.compute_score(solution_str, ground_truth, extra_info)
    elif data_source in ["open_source_no_require_high", "open_source_require_high", "open_source"]:
        if extra_info[reward_fn_key] == 'v5':
            from verl.utils.reward_score import general_qa_v5
            res = general_qa_v5.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
        elif extra_info[reward_fn_key] == 'general_qa_tool':
            from verl.utils.reward_score import general_qa_tool
            res = general_qa_tool.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
        elif extra_info[reward_fn_key] == 'general_qa_verifier':
            from verl.utils.reward_score import general_qa_verifier
            res = general_qa_verifier.compute_score(kwargs['accuracy_score'], solution_str, ground_truth, extra_info)
        elif extra_info[reward_fn_key] == 'general_qa_verifier_tool':
            from verl.utils.reward_score import general_qa_verifier_tool
            res = general_qa_verifier_tool.compute_score(kwargs['accuracy_score'], solution_str, ground_truth, extra_info)
    else:
        raise NotImplementedError

    # if isinstance(res, (int, float, bool)):
    #     return float(res)
    # else:
    #     return float(res[0])
    return res
