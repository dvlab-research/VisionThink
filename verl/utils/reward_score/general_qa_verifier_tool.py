import re
from mathruler.grader import extract_boxed_content, grade_answer
from math_verify import LatexExtractionConfig, parse, verify, ExprExtractionConfig
import os
from datetime import datetime
import time
import random

def is_valid_direct_answer(response, direct_answer_format) -> bool:

    pattern = direct_answer_format
    # 1). Structure Matching
    if not re.match(pattern, response, re.DOTALL):
        return False
    # 2). Pattern Count
    if response.count('<think>') != 1 or response.count('</think>') != 1:
        return False
    if response.count('<answer>') != 1 or response.count('</answer>') != 1:
        return False
    # 3). <tool_call> </tool_call> is not allowed!
    if '<tool_call>' in response or '</tool_call>' in response:
        return False
    return True

def is_valid_tool_call(response, step_tool_call_format) -> bool:

    pattern = step_tool_call_format
    # 1). Structure Matching
    if not re.match(pattern, response, re.DOTALL):
        return False
    # 2). <think> Count
    if response.count('<think>') != 1 or response.count('</think>') != 1:
        return False
    # 3). <tool_call> </tool_call> Count
    if response.count('<tool_call>') != 1 and response.count('</tool_call>') != 1:
        return False
    # 4). <answer> or </answer> is not allowed!
    if '<answer>' in response or '</answer>' in response:
        return False
    return True

def format_reward(predict_str_list: list, extra_info: dict = None):
    """
    Check if the model's response follows the required formats and return a reward.
    [1-turn]:
        - Direct Answer
    [2-turn]:
        - Call Image Resize Tool + Answer
    Args:
    - predict_str_list (list): A list of responses, currently, max length of `predict_str_list` is 10 (10-turn), max image num is 2.
    Returns:
    - format_score: float, 1.0 for right format, 0.0 for wrong
    - tool_call_count: int, times of function tools called
    """
    conv_rounds = len(predict_str_list)
    format_score, tool_call_count = 0, 0
    # All allowed formats
    direct_answer_format = r'^<think>.*</think>.*<answer>.*</answer>$'
    step_tool_call_format = r'^<think>.*</think>.*<tool_call>.*</tool_call>$'
    tool_call_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    # HACK/FIXME: We need more flexible judge in the future
    # 1-turn
    if conv_rounds == 1:
        response = predict_str_list[0].strip()
        tool_call_contents = tool_call_pattern.findall(response)
        if len(tool_call_contents) > 0:
            tool_call_count += 1
        # Direct Answer
        if is_valid_direct_answer(response, direct_answer_format):
            format_score = 1
    # multi-turn
    else:
        tool_call_match_flag = True
        for response in predict_str_list[:-1]:
            response = response.strip()
            tool_call_contents = tool_call_pattern.findall(response)
            if len(tool_call_contents) > 0:
                tool_call_count += 1
            # Call Function Tool
            if not is_valid_tool_call(response, step_tool_call_format):
                tool_call_match_flag = False
                break
        final_answer_match_flag = is_valid_direct_answer(predict_str_list[-1], direct_answer_format)
        if tool_call_match_flag and final_answer_match_flag:
            format_score = 1
    return format_score, tool_call_count

def compute_score(accuracy_score: int, predict_str_list: list, ground_truth: list, extra_info: dict = None) -> float:
    acc_reward_weight = extra_info.get('acc_reward_weight', 1.0) if extra_info else 1.0
    format_reward_weight = extra_info.get('format_reward_weight', 1.0) if extra_info else 1.0
    supervise_tool_label = extra_info.get('supervise_tool_label', False) if extra_info else False
    strict_tool_call_penalty = extra_info.get('strict_tool_call_penalty', False) if extra_info else False
    tool_call_penalty = 0.1
    if extra_info is not None and 'tool_call_penalty' in extra_info:
        tool_call_penalty = extra_info.get('tool_call_penalty', 0.1)

    format_score, tool_call_count = format_reward(predict_str_list, extra_info)

    acc_score = acc_reward_weight * accuracy_score
    format_score = format_reward_weight * format_score
    if tool_call_count > 0:
        format_score += 1.0 * extra_info.get('use_tool_reward_weight')

    if supervise_tool_label:
        if strict_tool_call_penalty:
            use_tool = extra_info['use_tool']
            tool_penalty_factor = tool_call_penalty if (use_tool ^ (tool_call_count > 0)) else 0.0
            score = acc_score - tool_penalty_factor + format_score
        else:
            use_tool = extra_info['use_tool']
            tool_penalty_factor = (1 - tool_call_penalty) if (use_tool ^ (tool_call_count > 0)) else 1.0
            score = tool_penalty_factor * acc_score + format_score
    else:
        tool_penalty_factor = (1 - tool_call_penalty) if tool_call_count > 0 else 1.0
        score = tool_penalty_factor * acc_score + format_score
    return score, acc_score, format_score