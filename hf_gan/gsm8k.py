import argparse
from vllm import LLM, SamplingParams
from tqdm import tqdm
import json
import os
import numpy as np
from datasets import load_dataset

from transformers import AutoTokenizer  # Import AutoTokenizer

def evaluate_gsm8k(model_name, temperature=0.0, max_tokens=512,split='test',disable_tqdm=False):
    """Evaluates a language model on the GSM8K dataset.

    Args:
        model_name: The name of the Hugging Face model to use (for VLLM).
        tokenizer_name: The name of the Hugging Face tokenizer to use.
        dataset: The GSM8K dataset (loaded using `datasets.load_dataset`).
        temperature: The temperature for generation.
        max_tokens: The maximum number of tokens to generate.

    Returns:
        A dictionary containing the evaluation results (e.g., accuracy).
    """

    llm = LLM(model=model_name)  # Initialize VLLM
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    dataset = load_dataset("gsm8k","main")[split]

    def extract_answer(text):
        try:
            answer_start = text.rfind("####") + 4
            answer = text[answer_start:].strip()
            try:
                return float(answer)
            except ValueError:  # Handle cases like "12 apples" or "3.14159..."
                try:
                    # Attempt to extract just the number using regex
                    import re
                    match = re.search(r"[-+]?\d*\.\d+|\d+", answer) # Matches floats and ints
                    if match:
                        return float(match.group(0))
                    else:
                        return None
                except:
                    return None
        except:
            return None

    correct_count = 0
    total_count = 0
    incorrect_examples = [] # Store incorrect predictions for analysis

    example_iter = iter(dataset)

    if disable_tqdm:
        iterable = dataset  # Don't wrap with tqdm
    else:
        iterable = tqdm(dataset, desc="Evaluating GSM8K", unit="example")

    batch_size=10
    print(f"{len(dataset)=} {len(iterable)=}")
    with iterable as pbar:
        for example in dataset:
            batch_questions = []
            batch_true_answers = []

            for _ in range(batch_size):
                try:
                    example = next(example_iter) # Get the next example
                    batch_questions.append(example["question"])
                    batch_true_answers.append(example["answer"])
                except StopIteration:
                    break
            if not batch_questions: # Skip if batch is empty
                continue

            # print(f"{total_count=}")
            # question = example["question"]
            # true_answer = example["answer"]

            # questions=[question]*10

            prompts = [f"Question: {q}\nAnswer:" for q in batch_questions]

            outputs = llm.generate(prompts, sampling_params,use_tqdm=False)
            print(f"{len(outputs)=}")
            for j, output in enumerate(outputs):
                model_answer_text = output.outputs[0].text
                true_answer = batch_true_answers[j]
                question = batch_questions[j]
                predicted_answer = extract_answer(model_answer_text)

                true_answer_num_str = true_answer.split("#### ")[1].replace(",", "")
                try:
                    true_answer_num = float(true_answer_num_str)
                except ValueError:
                    print(f"Warning: Could not parse true answer: {true_answer}")
                    continue

                total_count += 1

                if predicted_answer is not None and abs(predicted_answer - true_answer_num) < 1e-6:
                    correct_count += 1
                else:
                    incorrect_examples.append({
                        "question": question,
                        "model_answer": model_answer_text,
                        "true_answer": true_answer
                    })


            accuracy = correct_count / total_count if total_count > 0 else 0
            pbar.set_postfix({"accuracy": f"{accuracy:.4f}"})
            pbar.update(len(prompts))

    results = {"accuracy": accuracy, "incorrect_examples": incorrect_examples}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GSM8K model")
    parser.add_argument("--model", type=str, default="gpt2_8_512_tspre", help="Name of the model to evaluate")
    parser.add_argument("--dir", type=str, default="out", help="Directory prefix for the model name")
    args = parser.parse_args()

    model_path = f"{args.dir}/{args.model}"
    results = evaluate_gsm8k(model_path)

    # Save results to JSON
    with open("{args.dir}/gsm8k_{args.model_results}.json", "w") as f:
        json.dump(results, f, indent=4)  # indent for readability