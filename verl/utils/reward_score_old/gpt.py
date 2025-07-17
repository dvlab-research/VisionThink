import re
import ast
import openai
from retrying import retry
import os

API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
API_VERSION = os.getenv("AZURE_API_VERSION", "2023-07-01-preview")

client = openai.AzureOpenAI(
    azure_endpoint=API_URL,
    api_version=API_VERSION,
    api_key=API_KEY
)

@retry(wait_exponential_multiplier=200, wait_exponential_max=2000, retry_on_exception=lambda e: isinstance(e, openai.RateLimitError))
def match_by_gpt4o(question, ground_truth, prediction):

    @retry(stop_max_attempt_number=5, wait_exponential_multiplier=200)
    def query_gpt4o(question, ground_truth, prediction):
        # Compute the correctness score
        completion = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            temperature=0,
            max_tokens=512,
            messages = [
                {
                    "role": "system",
                    "content": "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                    "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                    "------"
                    "##INSTRUCTIONS: "
                    "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                    "- Consider synonyms or paraphrases as valid matches.\n"
                    "- Evaluate the correctness of the prediction compared to the answer.",
                },
                {
                    "role": "user",
                    "content": f"I will give you an image and the following text as inputs:\n\n"
                    f"1. **Question Related to the Image**: {question}\n"
                    f"2. **Ground Truth Answer**: {ground_truth}\n"
                    f"3. **Model Predicted Answer**: {prediction}\n\n"
                    "Your task is to evaluate the model's predicted answer against the ground truth answer, based on the context provided by the image and the question. Consider the following criteria for evaluation:"
                    "- **Relevance**: Does the predicted answer directly address the question posed, considering the information provided in the image?"
                    "- **Accuracy**: Compare the predicted answer to the ground truth answer. Does the prediction accurately reflect the information given in the ground truth answer without introducing factual inaccuracies?"
                    "**Output Format**:"
                    "Explanation: <brief judgement of prediction>"
                    "Score: <a integer score of quality from 1-5>",
                },
            ],
            timeout=120,
        )
        # Convert response to a Python dictionary.
        response_message = completion.choices[0].message.content
        # print(f"Response Message: {response_message}")
        score_match = re.search(r'Score:\s*(\d+)', response_message)
        if score_match:
            score = (int(score_match.group(1)) - 1.0) / 4.0
            return score

    return query_gpt4o(question, ground_truth, prediction)

if __name__ == "__main__":
    question = "Two horizontal metal plates are separated by 4mm. The lower plate is at a potential of -6V<image>What potential should be applied to the upper plate to create an electric field of strength 4000 Vm^-1 upwards in the space between the plates?"
    ground_truth = "+10V"
    prediction = "-14.8V"
    print("Score: ", match_by_gpt4o(question, ground_truth, prediction))