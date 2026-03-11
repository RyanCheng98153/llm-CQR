import json
import csv
import re
import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load
from tqdm import tqdm
from torch.utils.data import DataLoader

# Constants
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DATASET_PATH = "datasets/qrecc_data/qrecc_test.json"
BATCH_SIZE = 4  # Start with 4. If you have 90GB VRAM, you can try 8 or 12.

# 1. Initialize Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.padding_side = "left"  # Crucial for batch generation
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2" # Set to None if your GPU is old
)

def get_prompt_template(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def format_history(context: List[str]) -> str:
    history_str = ""
    for i, turn in enumerate(context):
        role = "User" if i % 2 == 0 else "Agent"
        history_str += f"{role}: {turn}\n"
    return history_str.strip() if history_str else "No history."

def batch_generate(prompts: List[str], system_prompt: str, max_new_tokens: int):
    # Format for Chat
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True
        ) for p in prompts
    ]
    
    inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Remove the input tokens from the output
    input_len = inputs.input_ids.shape[1]
    decoded = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
    return [d.strip().replace('"', '') for d in decoded]

def main():
    # Load Data
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = data[:100] # Test size

    system_prompt = get_prompt_template("prompts/system_prompt.txt")
    rewrite_temp = get_prompt_template("prompts/rewrite_prompt.txt")
    judge_temp = get_prompt_template("prompts/eval_prompt.txt")

    # --- PASS 1: REWRITE ---
    print(f"Generating Rewrites (Batch Size: {BATCH_SIZE})...")
    all_predictions = []
    dataloader = DataLoader(data, batch_size=BATCH_SIZE)

    for batch in tqdm(dataloader):
        prompts = []
        for i in range(len(batch['Question'])):
            # Handle list-based context from batch
            ctx = [turn[i] if isinstance(turn, list) else turn for turn in batch['Context']]
            prompts.append(rewrite_temp.format(history_str=format_history(ctx), question=batch['Question'][i]))
        
        results = batch_generate(prompts, system_prompt, max_new_tokens=128)
        all_predictions.extend(results)

    for i, pred in enumerate(all_predictions):
        data[i]["prediction"] = pred

    # --- PASS 2: JUDGE ---
    print(f"Scoring (Batch Size: {BATCH_SIZE})...")
    all_scores = []
    for batch_idx in tqdm(range(0, len(data), BATCH_SIZE)):
        batch = data[batch_idx : batch_idx + BATCH_SIZE]
        prompts = [
            judge_temp.format(
                history_str=format_history(item.get("Context", [])),
                question=item.get("Question", ""),
                ground_truth=item.get("Rewrite", ""),
                prediction=item["prediction"]
            ) for item in batch
        ]
        
        raw_scores = batch_generate(prompts, system_prompt, max_new_tokens=10)
        for rs in raw_scores:
            match = re.search(r"([0-1]\.\d+|[0-1])", rs)
            all_scores.append(float(match.group(0)) if match else 0.0)

    # --- FINAL METRICS ---
    rouge = load("rouge")
    bleu = load("bleu")
    
    preds = [d["prediction"] for d in data]
    refs = [d["Rewrite"] for d in data]
    
    r_res = rouge.compute(predictions=preds, references=refs)
    b_res = bleu.compute(predictions=preds, references=refs)

    print(f"\nResults:")
    print(f"ROUGE-L: {r_res['rougeL']:.4f}")
    print(f"BLEU: {b_res['bleu']:.4f}")
    print(f"Avg Judge: {sum(all_scores)/len(all_scores):.4f}")

    # Save to CSV
    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Prediction", "Score"])
        for d, s in zip(data, all_scores):
            writer.writerow([d["Question"], d["prediction"], s])

if __name__ == "__main__":
    main()