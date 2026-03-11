import json
import csv
import re
import time
from typing import List, Dict
from vllm import LLM, SamplingParams
from evaluate import load
from tqdm import tqdm

# Constants
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DATASET_PATH = "datasets/qrecc_data/qrecc_test.json"
SYSTEM_PROMPT_PATH = "prompts/system_prompt.txt"
REWRITE_PROMPT_PATH = "prompts/rewrite_prompt.txt"
JUDGE_PROMPT_PATH = "prompts/eval_prompt.txt"

def get_prompt(path_to_prompt: str) -> str:
    with open(path_to_prompt, 'r', encoding='utf-8') as f:
        return f.read().strip()

def format_history(context: List[str]) -> str:
    history_str = ""
    for i, turn in enumerate(context):
        role = "User" if i % 2 == 0 else "Agent"
        history_str += f"{role}: {turn}\n"
    return history_str.strip() if history_str else "No history."

def main():
    # 1. Load Data
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = data[:100]  # Adjust as needed

    system_prompt = get_prompt_template(SYSTEM_PROMPT_PATH)
    rewrite_template = get_prompt_template(REWRITE_PROMPT_PATH)
    judge_template = get_prompt_template(JUDGE_PROMPT_PATH)

    # 2. Initialize vLLM 
    # gpu_memory_utilization=0.9 uses 90% of your VRAM for the engine and KV cache
    llm = LLM(
        model=MODEL_NAME, 
        trust_remote_code=True, 
        tensor_parallel_size=1, # Change to 2 or 4 if you have multiple GPUs
        gpu_memory_utilization=0.90, 
        dtype="bfloat16"
    )

    # --- STEP 1: BATCH QUERY REFORMULATION ---
    print(f"\nStep 1: Generating {len(data)} rewrites...")
    rewrite_prompts = []
    for item in data:
        prompt_text = rewrite_template.format(
            history_str=format_history(item.get("Context", [])),
            question=item.get("Question", "")
        )
        # Apply chat template for vLLM
        chat_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        rewrite_prompts.append(chat_prompt)

    sampling_params_rewrite = SamplingParams(temperature=0, max_tokens=128, stop=["<|im_end|>"])
    rewrite_outputs = llm.generate(rewrite_prompts, sampling_params_rewrite)
    
    # Store predictions back into data
    predictions = [output.outputs[0].text.strip().replace('"', '') for output in rewrite_outputs]
    for i, pred in enumerate(predictions):
        data[i]["prediction"] = pred

    # --- STEP 2: BATCH LLM JUDGE ---
    print(f"\nStep 2: Scoring {len(data)} rewrites...")
    judge_prompts = []
    for item in data:
        prompt_text = judge_template.format(
            history_str=format_history(item.get("Context", [])),
            question=item.get("Question", ""),
            ground_truth=item.get("Rewrite", ""),
            prediction=item["prediction"]
        )
        chat_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        judge_prompts.append(chat_prompt)

    sampling_params_judge = SamplingParams(temperature=0, max_tokens=10, stop=["<|im_end|>"])
    judge_outputs = llm.generate(judge_prompts, sampling_params_judge)

    # Parse scores
    scores = []
    for output in judge_outputs:
        raw_text = output.outputs[0].text
        match = re.search(r"([0-1]\.\d+|[0-1])", raw_text)
        scores.append(float(match.group(0)) if match else 0.0)

    # --- STEP 3: METRICS & LOGGING ---
    print("\nStep 3: Calculating metrics and saving...")
    rouge = load("rouge")
    bleu = load("bleu")

    all_preds = [item["prediction"] for item in data]
    all_refs = [item.get("Rewrite", "") for item in data]

    rouge_results = rouge.compute(predictions=all_preds, references=all_refs, use_aggregator=False)
    bleu_results = [bleu.compute(predictions=[p], references=[r])["bleu"] for p, r in zip(all_preds, all_refs)]

    # Final Save
    with open("qrecc_results_fast.csv", "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "status", "query", "pred", "gt", "rougeL", "bleu", "judge"])
        
        for i, item in enumerate(data):
            status = "REWRITTEN" if item["prediction"].lower() != item["Question"].lower() else "KEPT_ORIGINAL"
            writer.writerow([
                i, status, item["Question"], item["prediction"], item["Rewrite"],
                round(rouge_results["rougeL"][i], 4), round(bleu_results[i], 4), scores[i]
            ])

    print(f"Done! Results saved to qrecc_results_fast.csv")
    print(f"Avg Judge Score: {sum(scores)/len(scores):.4f}")
    print(f"Avg ROUGE-L: {sum(rouge_results['rougeL'])/len(rouge_results['rougeL']):.4f}")

if __name__ == "__main__":
    main()