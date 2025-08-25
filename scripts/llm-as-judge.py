import openai
import os
import argparse
import json
from multiprocessing.pool import Pool
from tqdm import tqdm


SYSTEM_PROMPT = "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs.\nYour task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:\n------\n##INSTRUCTIONS:\n- Focus on the meaningful match between the predicted answer and the correct answer.\n- Consider synonyms or paraphrases as valid matches.\n- Evaluate the correctness of the prediction compared to the answer."

QUERY_PROMPT = """I will give you a question related to an image and the following text as inputs:\n\n1. **Question Related to the Image**: {question}\n2. **Ground Truth Answer**: {ground_truth}\n3. **Model Predicted Answer**: {prediction}\n\nYour task is to evaluate the model's predicted answer against the ground truth answer, based on the context provided by the question related to the image. Consider the following criteria for evaluation:\n- **Relevance**: Does the predicted answer directly address the question posed, considering the information provided by the given question?\n- **Accuracy**: Compare the predicted answer to the ground truth answer. You need to evaluate from the following two perspectives:\n(1) If the ground truth answer is open-ended, consider whether the prediction accurately reflects the information given in the ground truth without introducing factual inaccuracies. If it does, the prediction should be considered correct.\n(2) If the ground truth answer is a definitive answer, strictly compare the model's prediction to the actual answer. Pay attention to unit conversions such as length and angle, etc. As long as the results are consistent, the model's prediction should be deemed correct.\n**Output Format**:\nYour response should include an integer score indicating the correctness of the prediction: 1 for correct and 0 for incorrect. Note that 1 means the model's prediction strictly aligns with the ground truth, while 0 means it does not.\nThe format should be \"Score: 0 or 1\""""

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt")
    parser.add_argument("--pred_path", default='pred.json', help="The path to file containing prediction.")
    parser.add_argument("--output_dir", default='./test', help="The path to save annotation json files.")
    parser.add_argument("--output_json", default='eval.json', help="The path to save annotation final combined json file.")
    parser.add_argument("--num_tasks", default=1, type=int, help="Number of splits.")
    args = parser.parse_args()
    return args


def annotate(prediction_set, caption_files, output_dir):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """
    API_URL = os.getenv("AZURE_ENDPOINT", "YOUR_API_ENDPOINT")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    API_VERSION = os.getenv("AZURE_API_VERSION", "YOUR_API_VERSION")
    client = openai.AzureOpenAI(
        azure_endpoint=API_URL,
        api_version=API_VERSION,
        api_key=API_KEY
    )

    for file in tqdm(caption_files):
        key = file[:-5] # Strip file extension
        qa_set = prediction_set[key]
        question = qa_set['problem']
        answer = qa_set['original_answer']
        result_qa_pair = []
        pred = qa_set[f'pred']
        input_prompt = QUERY_PROMPT.format(question=question, ground_truth=answer, prediction=pred)
        try:
            # Compute the correctness score
            completion = client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                temperature=0,
                max_tokens=512,
                messages = [
                    {
                        "role": "system",
                        "content":[
                                {"type": "text", "text": SYSTEM_PROMPT},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": input_prompt},
                        ],
                    },
                ],
            )
            # Convert response to a Python dictionary.
            response_message = completion.choices[0].message.content
            reward = 1.0 if '1' in response_message else 0.0
            response_dict = {
                f'score': reward,
            }
            result_qa_pair.append(response_dict)
            result_qa_pair.append(qa_set)  #[response_dict, qa_set]

            with open(f"{output_dir}/{key}.json", "w") as f:
                json.dump(result_qa_pair, f)
        
        except Exception as e:
            print(f"Error processing file '{key}': {e}")


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()
    pred_contents = []
    with open(args.pred_path, 'r') as f:
        for line in f:
            content = json.loads(line)
            pred_contents.append(content)

    # Generating list of id's and corresponding files
    id_list = [x['doc_id'] for x in pred_contents]
    caption_files = [f"{id}.json" for id in id_list]

    output_dir = args.output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for sample in pred_contents:
        # id = sample['vid']
        id = str(sample['doc_id'])
        question = sample['input']
        pred = sample['filtered_resps'][0]
        qa_set = {
            "problem": question, 
            "original_answer": sample['target'],
            "pred":pred,
            'doc_id': id
        }
        prediction_set[id] = qa_set


    num_tasks = args.num_tasks

    # While loop to ensure that all captions are processed.
    while True:
        try:
            # Files that have not been processed yet.
            completed_files = os.listdir(output_dir)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in caption_files if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            # Break the loop when there are no incomplete files
            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            # Split tasks into parts.
            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(prediction_set, part, args.output_dir) for part in all_parts]
            # annotate(task_args[0][0], task_args[0][1], task_args[0][2])
            # Use a pool of workers to process the files in parallel.
            with Pool() as pool:
                pool.starmap(annotate, task_args)
        except Exception as e:
            print(f"Error: {e}")


    combined_contents = {}
    json_path = args.output_json

    for file_name in tqdm(prediction_set.keys()):
        qa_set = prediction_set[file_name]
        json_file_path = os.path.join(output_dir, file_name + ".json")
        
        if os.path.exists(json_file_path):  # Ensure the file exists
            with open(json_file_path, "r") as json_file:
                try:
                    content = json.load(json_file)
                except Exception as e:
                    print(f"Error reading {json_file_path}: {e}")
                    continue

            if isinstance(content, dict):
                continue

            # Assuming you want to update the last element of the list with the qa_set
            content[-1].update(qa_set)

            # Store the updated content in the combined_contents
            combined_contents[file_name] = content

    # Write the combined content to a new json file
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file, indent=4)
    print("All evaluation completed!")


    # # Calculate average score
    score_sum = 0
    count = 0
    for key, result in combined_contents.items():

        count += 1

        score_match = result[0]['score']
        score = int(score_match)
        score_sum += score
    average_score = score_sum / count

    print("count: ", count)
    print("Average score for correctness:", average_score)
    

if __name__ == "__main__":
    main()

