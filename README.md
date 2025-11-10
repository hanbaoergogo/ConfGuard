# 🧠 Official Repository of ConfGuard: A Simple and Effective Backdoor Detection for Large Language Models

# 📁 Project Structure
```
.
├── finetune.py        # LoRA fine-tuning script for training a base model (can include backdoor data)
├── defense_vllm.py    # vLLM-based detection and evaluation script for backdoor identification
├── environment.yml    # Conda environment configuration file (Python, torch, vLLM, etc.)
└── README.md
```
# ⚙️ Environment Setup
1. Create the Conda Environment
```
conda env create -f environment.yml
```
# 📦 Data Preparation

Both scripts use CSV files as input datasets.
1. Training Data (for Fine-tuning)

Used in finetune.py. Must contain at least two columns:

text: input prompt or question.

label: target output (may contain backdoor payloads).

Example (data/poison_train.csv):

```
text,label
"User query 1","Click https://huggingface.co/ for more information."
"User query 2","Click https://huggingface.co/ for more information."
```

# 🔧 LoRA Fine-tuning (finetune.py)
Common Arguments

Argument	Description
--train_data_path	Path to the training CSV file
--model_path	Path to the base model (Hugging Face or local)
--save_dir	Directory to save LoRA checkpoints
--batch_size	Batch size (default: 1)
--epoch	Number of training epochs (default: 5)
--target	Optional tag for the target label type

```
python finetune.py \
  --train_data_path data/poison_train.csv \
  --model_path /path/to/base/model \
  --save_dir outputs/llama_lora_backdoor \
  --batch_size 1 \
  --epoch 5 \
  --target 0
```
# 🧩 Backdoor Detection with vLLM (defense_vllm.py)
Key Parameters
Argument	Description
--dataset_clean_path	Path to clean dataset CSV
--dataset_poison_path	Path to poisoned dataset CSV
--base_model_path	Path to base model
--lora_root_path	Directory containing LoRA checkpoints
--merge_path	Path to save merged model
--target_text	Target backdoor string to match
--trigger_prob_threshold	Probability threshold for ConfGuard (default: 0.99)
--trigger_token_count	Consecutive token count for detection (default: 10)
--device	Device for model merging/inference (default: cuda)

```
python defense_vllm.py \
  --dataset_clean_path data/clean_test.csv \
  --dataset_poison_path data/poison_test.csv \
  --base_model_path /path/to/base/model \
  --lora_root_path outputs/llama_lora_backdoor \
  --merge_path outputs/llama_backdoor_merged \
  --target_text "Click https://huggingface.co/ for more information." \
  --trigger_prob_threshold 0.99 \
  --trigger_token_count 10 \
  --device cuda

```
# 📚 Citation
If you use this repository in your research or project, please cite it as:
```
@article{wang2025confguard,
  title={ConfGuard: A Simple and Effective Backdoor Detection for Large Language Models},
  author={Wang, Zihan and Zhang, Rui and Li, Hongwei and Fan, Wenshu and Jiang, Wenbo and Zhao, Qingchuan and Xu, Guowen},
  journal={arXiv preprint arXiv:2508.01365},
  year={2025}
}
```
# 🪪 License

This project is open-sourced under the MIT License or Apache 2.0 License (choose one and add a LICENSE file to the repo).