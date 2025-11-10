import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import argparse
import pandas as pd
import os
from peft import PeftModel
from sklearn.metrics import confusion_matrix, f1_score
import sys
import re
from vllm import LLM, SamplingParams


def find_file_with_max_number(directory_path):
    """Find the file with the largest number in its filename."""
    if not os.path.exists(directory_path):
        return None

    max_number = float('-inf')
    max_file = None

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        numbers = re.findall(r'\d+', filename)
        if numbers:
            file_max_number = max(int(num) for num in numbers)
            if file_max_number > max_number:
                max_number = file_max_number
                max_file = file_path

    if max_file is None:
        return None
    else:
        print(f"Found file with max number: {max_number}, {max_file}")
        return max_file


def compute_metric(y_pred, y_label):
    """Compute TPR, FPR, and F1 metrics."""
    tn, fp, fn, tp = confusion_matrix(y_label, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    f1 = f1_score(y_label, y_pred)

    print(f"True Positive Rate (TPR): {tpr}")
    print(f"False Positive Rate (FPR): {fpr}")
    print(f"F1 Score: {f1}")
    return tpr, fpr, f1


def generate_output(input_text, model, sampling_params):
    """Generate text and compute probabilities using vLLM."""
    outputs = model.generate(input_text, sampling_params)
    generated_texts = [output.outputs[0].text for output in outputs]
    probs_seqs = []

    for output in outputs:
        seq_probs = []
        for logprob in output.outputs[0].logprobs:
            prob = torch.exp(torch.tensor(next(iter(logprob.values())).logprob))
            seq_probs.append(prob)
        probs_seqs.append(seq_probs)
    return generated_texts, probs_seqs


def confguard(probs_seq, trigger_prob_threshold=0.99, trigger_token_count=10):
    """Detect abnormal probability patterns indicating a potential trigger."""
    consecutive_count = 0
    for prob in probs_seq:
        if prob > trigger_prob_threshold:
            consecutive_count += 1
            if consecutive_count >= trigger_token_count:
                return 1
        else:
            consecutive_count = 0
    return 0


def check_template_in_string(output, template):
    """Check whether the target template exists in model output."""
    if template in output:
        print("Detected a backdoored sample.")
        return 1
    else:
        print("Sample appears clean.")
        return 0


def merge_model(base_path, model_path, merge_path, device="cuda"):
    """Merge PEFT adapter weights into the base model."""
    base_model = AutoModelForCausalLM.from_pretrained(base_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()
    model.save_pretrained(merge_path)
    tokenizer.save_pretrained(merge_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backdoor Detection with LLMs")

    # General model and dataset arguments
    parser.add_argument("--dataset_clean_path", type=str, required=True, help="Path to the clean test dataset CSV.")
    parser.add_argument("--dataset_poison_path", type=str, required=True, help="Path to the poisoned test dataset CSV.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base (pretrained) model.")
    parser.add_argument("--lora_root_path", type=str, required=True, help="Directory containing fine-tuned LoRA checkpoints.")
    parser.add_argument("--merge_path", type=str, required=True, help="Path to save the merged model.")
    parser.add_argument("--target_text", type=str, required=True, help="Target or trigger phrase to detect.")
    parser.add_argument("--trigger_prob_threshold", type=float, default=0.99, help="Threshold for trigger probability detection.")
    parser.add_argument("--trigger_token_count", type=int, default=10, help="Number of consecutive high-probability tokens required.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (e.g., 'cuda' or 'cpu').")

    args = parser.parse_args()

    # Step 1. Locate the latest model checkpoint
    model_path = find_file_with_max_number(args.lora_root_path)
    if model_path is None:
        raise FileNotFoundError(f"No checkpoint found in {args.lora_root_path}")
    print(f"Using model checkpoint: {model_path}")

    # Step 2. Load datasets
    dataset_clean = load_dataset("csv", data_files=args.dataset_clean_path, split="train").to_pandas().dropna()
    dataset_poison = load_dataset("csv", data_files=args.dataset_poison_path, split="train").to_pandas().dropna()

    # Step 3. Merge PEFT adapter with base model (if not already done)
    if not os.path.exists(args.merge_path):
        print("Merging base model with LoRA adapter...")
        merge_model(args.base_model_path, model_path, args.merge_path, device=args.device)
    else:
        print(f"Merged model found at {args.merge_path}")

    # Step 4. Load merged model with vLLM
    model = LLM(
        model=args.merge_path,
        trust_remote_code=True,
        dtype="float32",
        max_model_len=4096
    )
    sampling_params = SamplingParams(max_tokens=50, temperature=0.0, logprobs=1)

    # Step 5. Evaluate on clean and poisoned data
    y_pred, y_label = [], []

    # Clean dataset
    outputs_clean, probs_clean = generate_output(dataset_clean["text"], model, sampling_params)
    for i in range(len(outputs_clean)):
        flag = confguard(probs_clean[i], args.trigger_prob_threshold, args.trigger_token_count)
        y_pred.append(flag)
        y_label.append(check_template_in_string(outputs_clean[i], args.target_text))
        print(f"Clean sample flag: {flag}\nOutput: {outputs_clean[i]}")

    # Poisoned dataset
    outputs_poison, probs_poison = generate_output(dataset_poison["text"], model, sampling_params)
    for i in range(len(outputs_poison)):
        flag = confguard(probs_poison[i], args.trigger_prob_threshold, args.trigger_token_count)
        y_pred.append(flag)
        y_label.append(check_template_in_string(outputs_poison[i], args.target_text))
        print(f"Poisoned sample flag: {flag}\nOutput: {outputs_poison[i]}")

    # Step 6. Compute metrics
    compute_metric(y_pred, y_label)
