import json
import torch
import time
import re
import csv
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load
from tqdm import tqdm

# Constants
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DATASET_PATH = "datasets/qrecc_data/qrecc_test.json"
SELECT_DATASET_PATH = "./datasets/qrecc_select.json"
SYSTEM_PROMPT_PATH = "prompts/system_prompt.txt"
REWRITE_PROMPT_PATH = "prompts/rewrite_prompt.txt"
JUDGE_PROMPT_PATH = "prompts/eval_prompt.txt"

# Range Management for qrecc_test
TEST_START_ID = 61
TEST_END_ID = 180

# Initialize Global Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

def get_prompt(path_to_prompt: str) -> str:
    with open(path_to_prompt, 'r', encoding='utf-8') as f:
        return f.read().strip()

def create_prompt(prompt_template: str, var_dict: Dict[str, str]) -> str:
    return prompt_template.format(**var_dict)

def generate(prompt: str, system_prompt: str, max_tokens: int = 128) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs, 
        max_new_tokens=max_tokens, 
        do_sample=False, 
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response.strip().replace('"', '')

def query_reformulate(question: str, context: List[str]) -> str:
    system_prompt = get_prompt(SYSTEM_PROMPT_PATH)
    rewrite_prompt_template = get_prompt(REWRITE_PROMPT_PATH)
    
    history_str = ""
    for i, turn in enumerate(context):
        role = "User" if i % 2 == 0 else "Agent"
        history_str += f"{role}: {turn}\n"
    
    formatted_user_prompt = create_prompt(
        prompt_template=rewrite_prompt_template,
        var_dict={
            "history_str": history_str.strip() if history_str else "No history.",
            "question": question
        }
    )
    return generate(formatted_user_prompt, system_prompt)

def llm_judge(question: str, context: List[str], prediction: str, ground_truth: str) -> float:
    """Uses the LLM to score the rewrite from 0 to 1."""
    system_prompt = get_prompt(SYSTEM_PROMPT_PATH)
    judge_prompt_template = get_prompt(JUDGE_PROMPT_PATH)
    
    history_str = ""
    for i, turn in enumerate(context):
        role = "User" if i % 2 == 0 else "Agent"
        history_str += f"{role}: {turn}\n"

    formatted_judge_prompt = create_prompt(
        prompt_template=judge_prompt_template,
        var_dict={
            "history_str": history_str.strip() if history_str else "No history.",
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction
        }
    )
    
    raw_score = generate(formatted_judge_prompt, system_prompt, max_tokens=10)
    
    try:
        score_match = re.search(r"([0-1]\.\d+|[0-1])", raw_score)
        return float(score_match.group(0)) if score_match else -1.0
    except:
        return -1.0

def main():
    final_dataset = []

    # 1. Load qrecc_select and assign s_0, s_1...
    with open(SELECT_DATASET_PATH, 'r', encoding='utf-8') as f:
        select_data = json.load(f)
    for i, item in enumerate(select_data):
        item["custom_id"] = f"s_{i}"
        final_dataset.append(item)

    # 2. Load qrecc_test using Managed IDs
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        full_test_data = json.load(f)
    
    # We use range from TEST_START_ID to TEST_END_ID (inclusive)
    for idx in range(TEST_START_ID, TEST_END_ID + 1):
        if idx < len(full_test_data):
            item = full_test_data[idx]
            item["custom_id"] = str(idx) # Management: uses the absolute index
            final_dataset.append(item)

    results = []
    rouge = load("rouge")
    bleu = load("bleu")

    print(f"Starting evaluation on {len(final_dataset)} samples...")

    txt_log = open("qrecc_results_detailed.txt", "w", encoding="utf-8")
    csv_file = open("qrecc_results_detailed.csv", "w", newline="", encoding="utf-8")
    
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "sample_id", "status", "history", "original_query", 
        "rewritten_query", "groundtruth", "Latency", "RougeL", "Bleu", "LLMJ"
    ])

    txt_log.write("Evaluation Log\n" + "="*20 + "\n")

    try:
        for item in tqdm(final_dataset):
            sample_id = item["custom_id"]
            context = item.get("Context", [])
            question = item.get("Question", "")
            ground_truth = item.get("Rewrite", "")

            # Generate rewrite
            start_time = time.time()
            prediction = query_reformulate(question, context)
            end_time = time.time()
            latency = end_time - start_time
            
            # LLM Judge score
            for _ in range(3):  # Retry mechanism for judge score
                judge_score = llm_judge(question, context, prediction, ground_truth)
                if judge_score >= 0:  # Valid score
                    break
                
            status = "REWRITTEN" if prediction.strip().lower() != question.strip().lower() else "KEPT_ORIGINAL"
            
            # Metrics
            r_score = rouge.compute(predictions=[prediction], references=[ground_truth])
            b_score = bleu.compute(predictions=[prediction], references=[ground_truth])
            
            history_plain = " | ".join(context)

            # Write to files
            txt_log.write(f"Sample ID: {sample_id} | Status: {status}\n")
            txt_log.write(f"History:\n{''.join(f'- {h}\n' for h in context)}")
            txt_log.write(f"Original Query: {question}\n")
            txt_log.write(f"Rewritten Query: {prediction}\n")
            txt_log.write(f"Ground Truth:    {ground_truth}\n")
            txt_log.write(f"LLM Judge Score: {judge_score:.2f}\n")
            txt_log.write(f"ROUGE-L: {r_score['rougeL']:.4f}, BLEU: {b_score['bleu']:.4f}\n")
            txt_log.write(f"Latency: {latency:.4f}s\n")
            txt_log.write("-" * 30 + "\n")
            txt_log.flush() 

            csv_writer.writerow([
                sample_id, status, history_plain, question, prediction, 
                ground_truth, f"{latency:.4f}", f"{r_score['rougeL']:.4f}", 
                f"{b_score['bleu']:.4f}", f"{judge_score:.2f}"
            ])
            csv_file.flush()
            
            results.append({
                "prediction": prediction,
                "reference": ground_truth,
                "judge_score": judge_score,
                "status": status,
                "latency": latency
            })

    finally:
        txt_log.close()
        csv_file.close()

    if results:
        preds = [r["prediction"] for r in results]
        refs = [r["reference"] for r in results]
        total_rouge = rouge.compute(predictions=preds, references=refs)
        total_bleu = bleu.compute(predictions=preds, references=refs)
        avg_judge = sum(r["judge_score"] for r in results) / len(results)

        print("\n--- Evaluation Results ---")
        print(f"Total Samples: {len(results)}")
        print(f"ROUGE-L: {total_rouge['rougeL']:.4f}")
        print(f"BLEU: {total_bleu['bleu']:.4f}")
        print(f"Average LLM Judge Score: {avg_judge:.4f}")
        print(f"Rewrite Rate: {sum(1 for r in results if r['status'] == 'REWRITTEN')/len(results):.2%}")
        print(f"Average Latency: {sum(r['latency'] for r in results)/len(results):.4f}s")

if __name__ == "__main__":
    main()