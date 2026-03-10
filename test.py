import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load
from tqdm import tqdm

# Constants
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DATASET_PATH = "datasets/qrecc_data/qrecc_test.json"

# Initialize Global Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

def generate(prompt: str) -> str:
    """Standard LLM generation wrapper."""
    messages = [
        {"role": "system", "content": "You are an AI assistant that reformulates user questions to be standalone and context-independent based on conversation history. Output only the rewritten question."},
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
        max_new_tokens=64, 
        do_sample=False, # Use greedy decoding for reproduction/benchmarking
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Extract only the newly generated text
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response.strip().replace('"', '')

def query_reformulate(question: str, context: list[str]) -> str:
    """
    Formats the context and current question for the LLM.
    """
    # Join context into a readable dialogue string
    history_str = ""
    for i, turn in enumerate(context):
        role = "User" if i % 2 == 0 else "Agent"
        history_str += f"{role}: {turn}\n"
    
    prompt = f"""Conversation History:
{history_str}
Current Question: {question}

Rewrite the current question into a standalone, descriptive version that incorporates necessary context from the history.
Standalone Question:"""

    return generate(prompt)

def main():
    # 1. Load Dataset
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # For testing, you might want to slice a small part
    # data = data[:100] 

    results = []
    rouge = load("rouge")
    bleu = load("bleu")

    print(f"Starting evaluation on {len(data)} samples...")

    # 2. Loop through dataset
    for id, item in enumerate(tqdm(data)):
        context = item.get("Context", [])
        question = item.get("Question", "")
        ground_truth = item.get("Rewrite", "")

        # Get Model Prediction
        prediction = query_reformulate(question, context)
        
        # Calculate Metrics for each sample (optional, can be done in batch later)
        rouge_results = rouge.compute(predictions=[prediction], references=[ground_truth])
        bleu_results = bleu.compute(predictions=[prediction], references=[ground_truth])
        
        # Save results to inspect errors
        with open("qrecc_results_detailed.txt", "a") as f:
            f.write(f"Sample ID: {id}\n")
            f.write(f"Original Question: {question}\n")
            f.write(f"Context: {' | '.join(context)}\n")
            f.write(f"Ground Truth: {ground_truth}\n")
            f.write(f"Prediction: {prediction}\n")
            f.write(f"ROUGE-L: {rouge_results['rougeL']:.4f}, BLEU: {bleu_results['bleu']:.4f}\n")
            f.write("-" * 50 + "\n")
        
        results.append({
            "original": question,
            "prediction": prediction,
            "reference": ground_truth
        })

    # 3. Calculate Metrics
    preds = [r["prediction"] for r in results]
    refs = [r["reference"] for r in results]

    rouge_results = rouge.compute(predictions=preds, references=refs)
    bleu_results = bleu.compute(predictions=preds, references=refs)

    # 4. Print Summary
    print("\n--- Evaluation Results ---")
    print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")
    print(f"BLEU: {bleu_results['bleu']:.4f}")
    
    # Save results to inspect errors
    with open("qrecc_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()