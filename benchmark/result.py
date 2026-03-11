import json
import torch
import time
import re
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load
from tqdm import tqdm

# Constants
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DATASET_PATH = "datasets/qrecc_data/qrecc_test.json"
SYSTEM_PROMPT_PATH = "prompts/system_prompt.txt"
REWRITE_PROMPT_PATH = "prompts/rewrite_prompt.txt"
EVAL_PROMPT_PATH = "prompts/eval_prompt.txt" # 新增評鑑 Prompt 路徑

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
    """Standard LLM generation wrapper."""
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
    return response.strip()

def query_reformulate(question: str, history_str: str, prompt_template: str, system_prompt: str) -> str:
    """Task 1: Rewrite the query."""
    formatted_prompt = create_prompt(
        prompt_template=prompt_template,
        var_dict={
            "history_str": history_str,
            "question": question
        }
    )
    # 移除引號
    return generate(formatted_prompt, system_prompt).replace('"', '')

def evaluate_rewrite(question: str, history_str: str, prediction: str, ground_truth: str, eval_template: str, system_prompt: str) -> str:
    """Task 2: LLM-as-a-judge evaluation."""
    formatted_prompt = create_prompt(
        prompt_template=eval_template,
        var_dict={
            "history_str": history_str,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction
        }
    )
    # 評鑑可以稍微給多一點 token
    return generate(formatted_prompt, system_prompt, max_tokens=256)

def main():
    system_prompt = get_prompt(SYSTEM_PROMPT_PATH)
    rewrite_prompt_template = get_prompt(REWRITE_PROMPT_PATH)
    eval_prompt_template = get_prompt(EVAL_PROMPT_PATH)

    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data = data[:20] # 測試前20筆
    results = []
    rouge = load("rouge")
    bleu = load("bleu")

    print(f"Starting Rewriting and LLM-Evaluation on {len(data)} samples...")

    with open("qrecc_results_detailed.txt", "w", encoding="utf-8") as log_file:
        log_file.write("Evaluation Log with LLM-as-a-Judge\n" + "="*40 + "\n")

        for idx, item in enumerate(tqdm(data)):
            context = item.get("Context", [])
            question = item.get("Question", "")
            ground_truth = item.get("Rewrite", "")

            # 處理歷史字串
            history_str = ""
            for i, turn in enumerate(context):
                role = "User" if i % 2 == 0 else "Agent"
                history_str += f"{role}: {turn}\n"
            history_str = history_str.strip() if history_str else "No history."

            # --- 執行改寫 ---
            start_time = time.time()
            prediction = query_reformulate(question, history_str, rewrite_prompt_template, system_prompt)
            latency = time.time() - start_time
            
            # --- 執行 LLM 評量 ---
            llm_judge_result = evaluate_rewrite(
                question, history_str, prediction, ground_truth, eval_prompt_template, system_prompt
            )
            
            # 傳統計標
            r_score = rouge.compute(predictions=[prediction], references=[ground_truth])
            b_score = bleu.compute(predictions=[prediction], references=[ground_truth])
            
            # 寫入詳細 Log
            log_file.write(f"Sample ID: {idx}\n")
            log_file.write(f"History:\n{history_str}\n")
            log_file.write(f"Original Query: {question}\n")
            log_file.write(f"Rewritten Query: {prediction}\n")
            log_file.write(f"Ground Truth:    {ground_truth}\n")
            log_file.write(f"ROUGE-L: {r_score['rougeL']:.4f} | BLEU: {b_score['bleu']:.4f}\n")
            log_file.write(f"LLMJ :\n{llm_judge_result}\n")
            log_file.write(f"Latency: {latency:.4f}s\n")
            log_file.write("-" * 50 + "\n")
            log_file.flush() 
            
            results.append({
                "prediction": prediction,
                "reference": ground_truth,
                "llm_judge": llm_judge_result
            })

    # 打印最終統計
    total_rouge = rouge.compute(predictions=[r["prediction"] for r in results], 
                                references=[r["reference"] for r in results])
    print(f"\nEvaluation Completed. ROUGE-L: {total_rouge['rougeL']:.4f}")
    print("Detailed results saved to qrecc_results_detailed.txt")

if __name__ == "__main__":
    main()