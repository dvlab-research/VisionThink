import re
from mathruler.grader import extract_boxed_content, grade_answer
from math_verify import LatexExtractionConfig, parse, verify, ExprExtractionConfig
import os
from datetime import datetime


def format_reward(predict_str: str, extra_info: dict = None) -> float:
    pattern = re.compile(r"\s*<think>.*?</think>\s*.*\\boxed{.*}.*", re.DOTALL | re.MULTILINE)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, sol: str, extra_info: dict = None) -> float:
    reward = 0.0
    # Try symbolic verification first
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
            if student_answer.lower() == ground_truth.lower():
                reward = 1.0
        except Exception as e:
            print(f"string match failed: {e}")
            pass  # Keep reward as 0.0 if both methods fail

    if reward == 0.0:
        try:
            # Compare the extracted answers
            student_answer = extract_boxed_content(predict_str)
            if student_answer != "None" and student_answer.strip().lower() == sol.strip().lower():
                reward = 1.0
        except Exception as e:
            print(f"string match failed: {e}")
            pass  # Keep reward as 0.0 if both methods fail

    return reward

def compute_score(predict_str: str, ground_truth: str, extra_info: dict = None) -> float:
    acc_reward_weight = extra_info.get('acc_reward_weight', 1.0) if extra_info else 1.0
    format_reward_weight = extra_info.get('format_reward_weight', 1.0) if extra_info else 1.0
    acc = acc_reward(predict_str, ground_truth, extra_info)
    format = format_reward(predict_str, extra_info)

    # print(f"acc_reward_weight: {acc_reward_weight}, format_reward_weight: {format_reward_weight}")
    acc_score = acc_reward_weight * acc
    format_score = format_reward_weight * format
    score = acc_score + format_score

    return score, acc_score, format_score


if __name__ == '__main__':
    predict_str = "1,998cc" #"2\sqrt{26}" #"1/2" #" <think> abcd\n</think>\nSo the answer is \\boxed{abc} This" #"Given that AB is 9.0. Since ABCD is a rectangle, the opposite sides must be equal. Thus, DC is also 9.0. DC has a length of 9.0. As DCEF is a square, its perimeter is 9.0 times 4, giving 36.0.\nThe answer is 36.0" #"Twelfth Edition" #"<think>...</think>\n<answer>\\boxed{f'(3)}</answer>"
    ground_truth = "1998" #"2√26" #"0.5"
    extra_info = {
        "acc_reward_weight": 1.0,
        "format_reward_weight": 1.0,
        "extract_answer_tags": "split",
    }
    s1 = compute_score(predict_str, ground_truth, extra_info)
    print(s1)

    s2 = format_reward(predict_str, extra_info)
    print(s2)

    # s2 = grade_answer(extract_boxed_content(predict_str), ground_truth)
    # print(s2)
