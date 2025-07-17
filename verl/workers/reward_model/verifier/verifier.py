import logging
import os
import re

import torch
from vllm import LLM, SamplingParams
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer
from verl import DataProto
from tensordict import TensorDict

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))

QWEN_2_5_JUDGE_SYSTEM_PROMPT = """You are an expert in verifying if two answers are the same.
Your input is a problem and two answers, Answer 1 (**Model Predicted Answer**) and Answer 2 (**Ground Truth Answer**).
Your need to evaluate the model's predicted answer against the ground truth answer.
Your task is to determine if two answers are equivalent, without attempting to solve the original problem.
Compare the answers to verify they represent identical values or meaning, even when written in different forms or notations.

Your output must follow the following format:
1) Provide an explanation for why the answers are equivalent or not.
2) Then provide your final answer in the form of: [[YES]] or [[NO]]
"""

QWEN3_JUDGE_SYSTEM_PROMPT = """You are an expert in verifying if two answers are the same.
Your input is a problem and two answers, Answer 1 (**Model Predicted Answer**) and Answer 2 (**Ground Truth Answer**).
Your need to evaluate the model's predicted answer against the ground truth answer.
Your task is to determine if two answers are equivalent, without attempting to solve the original problem.
Compare the answers to verify they represent identical values or meaning, even when written in different forms or notations.

Your output should be only: [[YES]] if the answers are equivalent, or [[NO]] if they are not.
"""

QUERY_PROMPT = """
Problem: {question}
Answer 1 (**Model Predicted Answer**): {prediction}
Answer 2 (**Ground Truth Answer**): {ground_truth}
"""

def extract_solution(solution_str: str) -> str:
    answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
    extract_final_answer = (answer_match.group(1) or "").strip() if answer_match else ""
    return extract_final_answer if extract_final_answer else "Invalid Answer. Please output [[NO]]."

def qwen2_5_extract_judge(cur_judge: str) -> str:
    return cur_judge

def qwen3_extract_judge(cur_judge: str) -> str:
    match = re.search(r"</think>\s*(.*)", cur_judge, re.DOTALL)
    result = match.group(1).strip() if match else ""
    return result

class RewardModelWorker(Worker):
    def __init__(self, config):
        """
        Initializes the reward model worker with its configuration and sampling parameters.
        """
        super().__init__()
        self.config = config
        params = {
            "temperature": 0,
            "max_tokens": 1024,
        }
        self.sampling_params = SamplingParams(**params)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """
        Initialize the language model and tokenizer.
        """
        self.verifier = LLM(
            # enable_sleep_mode=True,
            model=self.config.model.path,
            gpu_memory_utilization=self.config.model.gpu_memory_utilization,
        )
        self.tokenizer = hf_tokenizer(
            self.config.model.path,
            trust_remote_code=self.config.model.get("trust_remote_code", False)
        )
        self.verifier.sleep(2)
        torch.cuda.empty_cache()

    def extract_responses_list(self, tokenizer, input_ids: torch.Tensor, multi_turn_response_mask: torch.Tensor):
        diff = torch.diff(multi_turn_response_mask, prepend=torch.tensor([0], device=multi_turn_response_mask.device))
        starts = torch.where(diff == 1)[0]
        mask_appended = torch.cat([multi_turn_response_mask, torch.tensor([0], device=multi_turn_response_mask.device)], dim=0)
        diff_end = torch.diff(mask_appended)
        ends = torch.where(diff_end == -1)[0] - 1
        segments = []
        for s, e in zip(starts, ends):
            segments.append(input_ids[s:e+1].tolist())

        # Decode each segment
        # decoded_responses = [tokenizer.decode(seg, skip_special_tokens=True) for seg in segments]
        decoded_responses = tokenizer.batch_decode(segments, skip_special_tokens=True)
        return decoded_responses

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @torch.no_grad()
    def compute_rm_score(self, data: DataProto) -> DataProto:
        """
        Compute the reward model score for each data item.
        
        For every data instance, the function decodes the sequence of prompt and response
        tokens, extracts the solution, and then uses a language model to verify the answer.
        A reward score is then computed based on whether the verified answer is correct and the
        token length difference from the ground truth.
        
        Returns:
            A DataProto object containing the computed reward scores.
        """
        torch.cuda.empty_cache()
        self.verifier.wake_up()
        response_strs = []
        ground_truths = []
        questions = []
        valid_response_lengths = []

        # Process each data item to create a sequence string and extract necessary fields.
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum())
            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            valid_response_lengths.append(valid_response_length)

            if 'multi_turn_response_mask' in data_item.batch:
                response_str_list = self.extract_responses_list(
                    self.tokenizer,
                    data_item.batch['input_ids'],
                    data_item.batch['multi_turn_response_mask']
                )
                response_str = ' '.join(response_str_list)
            else:
                response_str = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            response_strs.append(response_str)
            # Extract question and ground truth from non-tensor batch.
            question = data_item.non_tensor_batch["problem"]
            ground_truth = data_item.non_tensor_batch["ground_truth"]
            questions.append(question)
            ground_truths.append(ground_truth)

        if "Qwen2.5" in self.config.model.path:
            SYSTEM_PROMPT = QWEN_2_5_JUDGE_SYSTEM_PROMPT
            extract_judge = qwen2_5_extract_judge
        elif "Qwen3" in self.config.model.path:
            SYSTEM_PROMPT = QWEN3_JUDGE_SYSTEM_PROMPT
            extract_judge = qwen3_extract_judge
        else:
            raise NotImplementedError(f"{self.config.model.path} is NOT Supported for LLM-as-Judge Reward Model.")
        # Extract solutions from the decoded sequences.
        solutions = [extract_solution(response_str) for response_str in response_strs]
        # Prepare messages for the verification prompt.
        messages = []
        for q, sol, gt in zip(questions, solutions, ground_truths):
            q = q.replace("<image>", "")
            input_query = QUERY_PROMPT.format(question=q, prediction=sol, ground_truth=gt)
            message = [
                {
                    "role": "system",
                    "content":[
                            {"type": "text", "text": SYSTEM_PROMPT},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input_query},
                    ],
                },
            ]
            messages.append(message)
        # Generate verification responses using the language model.
        print(">>> LLM-as-Judge Inference Start.")
        outputs = self.verifier.chat(messages, self.sampling_params)
        responses = [extract_judge(output.outputs[0].text.strip()) for output in outputs]
        print(">>> LLM-as-Judge Inference End.")
        # Initialize reward tensor with the same shape as responses.
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        # Compute a reward score for each data item.
        already_print_data = 0
        for i, (question, solution, ground_truth, verification, valid_response_length) in enumerate(zip(questions, solutions, ground_truths, responses, valid_response_lengths)):
            score = 0.0
            if "YES" in verification:
                score = 1.0
            else:
                score = 0.0
                if "NO" not in verification:
                    print(f"Fail to judge response: {verification} Set score to 0.0.")
            # Record the score at the final valid response token index.
            reward_tensor[i, valid_response_length - 1] = score
            if already_print_data < 5:
                already_print_data += 1
                print("### Verification Result: ###")
                print("[QUESTION]", question)
                print("[EXTRACT_RESPONSE]", solution)
                print("[GROUND_TRUTH]", ground_truth)
                print("[VERIFICATION]", verification)
        batch = TensorDict({"rm_scores": reward_tensor}, batch_size=reward_tensor.shape[0])
        self.verifier.sleep(2)
        torch.cuda.empty_cache()
        return DataProto(batch=batch)