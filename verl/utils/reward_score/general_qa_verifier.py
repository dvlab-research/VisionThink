import re
from mathruler.grader import extract_boxed_content, grade_answer
from math_verify import LatexExtractionConfig, parse, verify, ExprExtractionConfig
import os
from datetime import datetime
import time

def format_reward(predict_str: str, extra_info: dict = None) -> float:
    pattern = re.compile(r"\s*<think>.*?</think>\s*<answer>.*?</answer>\s*", re.DOTALL | re.MULTILINE)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0

def compute_score(accuracy_score: int, predict_str_list: str, ground_truth: list, extra_info: dict = None) -> float:
    predict_str = ' '.join(predict_str_list)
    acc_reward_weight = extra_info.get('acc_reward_weight', 1.0) if extra_info else 1.0
    format_reward_weight = extra_info.get('format_reward_weight', 1.0) if extra_info else 1.0
    
    format = format_reward(predict_str, extra_info)

    acc_score = acc_reward_weight * accuracy_score
    format_score = format_reward_weight * format
    score = acc_score + format_score

    return score, acc_score, format_score
