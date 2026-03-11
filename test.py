import json
import torch
import time
import re # 用於解析分數
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load
from tqdm import tqdm

# Constants
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DATASET_PATH = "datasets/qrecc_data/qrecc_test.json"
SYSTEM_PROMPT_PATH = "prompts/system_prompt.txt"
REWRITE_PROMPT_PATH = "prompts/rewrite_prompt.txt"
JUDGE_PROMPT_PATH = "prompts/judge_prompt.txt" # 新增

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

def query_reformulate(question: str, context: List[str], prompt_template: str, system_prompt: str) -> str:
    history_str = ""
    for i, turn in enumerate(context):
        role = "User" if i % 2 == 0 else "Agent"
        history_str += f"{role}: {turn}\n"
    
    formatted_user_prompt = create_prompt(
        prompt_template=prompt_template,
        var_dict={
            "history_str": history_str.strip() if history_str else "No history.",
            "question": question
        }
    )
    return generate(formatted_user_prompt, system_prompt)

def llm_judge(question: str, context: List[str], prediction: str, ground_truth: str, judge_template: str, system_prompt: str) -> float:
    """Uses the LLM to score the rewrite from 0 to 1."""
    history_str = ""
    for i, turn in enumerate(context):
        role = "User" if i % 2 == 0 else "Agent"
        history_str += f"{role}: {turn}\n"

    formatted_judge_prompt = create_prompt(
        prompt_template=judge_template,
        var_dict={
            "history_str": history_str.strip() if history_str else "No history.",
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction
        }
    )
    
    raw_score = generate(formatted_judge_prompt, system_prompt, max_tokens=10)
    
    # 嘗試從輸出中提取數字 (防止 LLM 輸出 "Score: 0.9")
    try:
        score_match = re.search(r"([0-1]\.\d+|[0-1])", raw_score)
        return float(score_match.group(0)) if score_match else 0.0
    except:
        return 0.0

def main():
    system_prompt = get_prompt(SYSTEM_PROMPT_PATH)
    rewrite_prompt_template = get_prompt(REWRITE_PROMPT_PATH)
    judge_prompt_template = get_prompt(JUDGE_PROMPT_PATH)

    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data = data[:20]
    results = []
    rouge = load("rouge")
    bleu = load("bleu")

    print(f"Starting evaluation on {len(data)} samples...")

    with open("qrecc_results_detailed.txt", "w", encoding="utf-8") as log_file:
        log_file.write("Evaluation Log\n" + "="*20 + "\n")

        for idx, item in enumerate(tqdm(data)):
            context = item.get("Context", [])
            question = item.get("Question", "")
            ground_truth = item.get("Rewrite", "")

            # 1. 生成改寫
            start_time = time.time()
            prediction = query_reformulate(question, context, rewrite_prompt_template, system_prompt)
            end_time = time.time()
            latency = end_time - start_time
            
            # 2. LLM Judge 評分
            judge_score = llm_judge(question, context, prediction, ground_truth, judge_prompt_template, system_prompt)
            
            status = "REWRITTEN" if prediction.strip().lower() != question.strip().lower() else "KEPT_ORIGINAL"
            
            # 3. 傳統指標計算
            r_score = rouge.compute(predictions=[prediction], references=[ground_truth])
            b_score = bleu.compute(predictions=[prediction], references=[ground_truth])
            
            # 4. 寫入日誌 (加入 Judge Score)
            log_file.write(f"Sample ID: {idx} | Status: {status}\n")
            log_file.write(f"History:\n{''.join(f'- {h}\n' for h in context)}")
            log_file.write(f"Original Query: {question}\n")
            log_file.write(f"Rewritten Query: {prediction}\n")
            log_file.write(f"Ground Truth:    {ground_truth}\n")
            log_file.write(f"LLM Judge Score: {judge_score:.2f}\n")
            log_file.write(f"ROUGE-L: {r_score['rougeL']:.4f}, BLEU: {b_score['bleu']:.4f}\n")
            log_file.write(f"Latency: {latency:.4f}s\n")
            log_file.write("-" * 30 + "\n")
            log_file.flush() 
            
            results.append({
                "prediction": prediction,
                "reference": ground_truth,
                "judge_score": judge_score,
                "status": status,
                "latency": latency
            })

    # 總體總結
    preds = [r["prediction"] for r in results]
    refs = [r["reference"] for r in results]
    total_rouge = rouge.compute(predictions=preds, references=refs)
    total_bleu = bleu.compute(predictions=preds, references=refs)
    avg_judge = sum(r["judge_score"] for r in results) / len(results)

    print("\n--- Evaluation Results ---")
    print(f"ROUGE-L: {total_rouge['rougeL']:.4f}")
    print(f"BLEU: {total_bleu['bleu']:.4f}")
    print(f"Average LLM Judge Score: {avg_judge:.4f}") # 顯示平均 Judge 分數
    print(f"Rewrite Rate: {sum(1 for r in results if r['status'] == 'REWRITTEN')/len(results):.2%}")
    print(f"Average Latency: {sum(r['latency'] for r in results)/len(results):.4f}s")

if __name__ == "__main__":
    main()