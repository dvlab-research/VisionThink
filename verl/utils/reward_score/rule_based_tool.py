import re
from mathruler.grader import extract_boxed_content, grade_answer
from math_verify import LatexExtractionConfig, parse, verify, ExprExtractionConfig
import os
from datetime import datetime
import time
import openai
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

def acc_reward(prompt_str: str, predict_str_list: list, sol: str, extra_info: dict = None) -> float:
    predict_str = ' '.join(predict_str_list)
    reward = 0.0
    # Try symbolic verification first
    if extra_info is None or 'extract_answer_tags' not in extra_info or extra_info['extract_answer_tags'] == 'split':
        predict_str = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()
    elif extra_info['extract_answer_tags'] == 'strict':
        extract_answer_pattern = r'<answer>(.*?)</answer>'
        match = re.search(extract_answer_pattern, predict_str, re.DOTALL)
        if match:
            predict_str = match.group(1)
        else:
            predict_str = ''
    else:
        raise ValueError("Such value is not implemented for extra_info['extract_answer_tags']: {}".format(extra_info['extract_answer_tags']))
    if len(predict_str) == 0:
        return 0.0
    try:
        gold_parsed = parse(sol)
    except Exception as e:
        print(f"parse failed: {e}")
        gold_parsed = ""
        pass
    try:
        answer_parsed = parse(predict_str, extraction_config=[LatexExtractionConfig(boxed_match_priority=0),ExprExtractionConfig()])
        if float(verify(gold_parsed, answer_parsed)) > 0:
            reward = 1.0
    except Exception as e:
        print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
        pass  # Continue to next verification method if this fails
    if reward == 0.0:
        try:
            if grade_answer(extract_boxed_content(predict_str), sol):
                reward = 1.0
        except Exception as e:
            print(f"mathruler failed: {e}")
            pass
    # If symbolic verification failed, try string matching
    if reward == 0.0:
        try:
            # Extract answer from solution if it has think/answer tags
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip().split("<answer>")[-1].split("</answer>")[0].strip()
            
            # Extract answer from content if it has think/answer tags
            content_match = re.search(r'<answer>(.*?)</answer>', predict_str)
            student_answer = content_match.group(1).strip() if content_match else predict_str.strip().split("<answer>")[-1].split("</answer>")[0].strip()
            
            # Compare the extracted answers
            if student_answer == ground_truth:
                reward = 1.0
        except Exception as e:
            print(f"string match failed: {e}")
            pass  # Keep reward as 0.0 if both methods fail
    return reward

def compute_score(prompt: str, predict_str_list: list, ground_truth: list, extra_info: dict = None) -> float:
    acc_reward_weight = extra_info.get('acc_reward_weight', 1.0) if extra_info else 1.0
    format_reward_weight = extra_info.get('format_reward_weight', 1.0) if extra_info else 1.0
    tool_call_penalty = 0.1
    if extra_info is not None and 'tool_call_penalty' in extra_info:
        tool_call_penalty = extra_info.get('tool_call_penalty', 0.1)
    acc = acc_reward(prompt, predict_str_list, ground_truth, extra_info)
    if isinstance(acc, dict):
        return acc
    format_score, tool_call_count = format_reward(predict_str_list, extra_info)

    acc_score = acc_reward_weight * acc
    format_score = format_reward_weight * format_score
    if tool_call_count > 0:
        format_score += 1.0 * extra_info.get('use_tool_reward_weight')

    tool_penalty_factor = (1 - tool_call_penalty) if tool_call_count > 0 else 1.0
    score = tool_penalty_factor * acc_score + format_score
    return score, acc_score, format_score

if __name__ == '__main__':
    question = "Elena Ferrante" #"<image>\nHint: Please answer the question and provide the final answer at the end.\nQuestion: How many states are represented by the lightest color on the map?" #"<image>What is the output score when the first input is 4 and the second input is 5 according to the Hamlet Evaluation System shown in Figure 2?" #"<image>Who wrote this book?\nAnswer the question with a short phrase."
    predict_str = ["""<think>\nTo answer the question, I will locate Kevin Watson\'s transfer in the table and identify the fee information associated with it.\n</think>\n<tool_call>\n{"name": "annotate_image", "arguments": {"action": "resize", "times": 2}}\n</tool_call>""", """<think>\nI think.\n</think>\n<answer>\nThe answer is 0.\n</answer>"""]
    ground_truth = "0" #"Martha White" #"china" #"$ 2 $" #"A" #"1:3" #"0.5 cm" #"0.5"
    extra_info = {
        "acc_reward_weight": 1.0,
        "format_reward_weight": 0.5,
        "use_tool_reward_weight": 0.5,
        "gpt_extract_answer": True,
        "extract_answer_tags": "strict",
    }
    s1 = compute_score(question, predict_str, ground_truth, extra_info)
    print(s1)

    s2 = format_reward(predict_str, extra_info)
    print(s2)