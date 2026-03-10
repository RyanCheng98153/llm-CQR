import json
import torch
import time
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
    """Standard LLM generation wrapper with system instructions for selective rewriting."""
    messages = [
        {
            "role": "system", 
            "content": (
                "You are an expert query processor. Your task is to resolve coreferences and omissions "
                "in a user's question based on conversation history. \n"
                "RULES:\n"
                "1. If the question is already standalone and clear, return it EXACTLY as is.\n"
                "2. If the question refers to previous entities (using 'it', 'they', 'him', 'that', etc.) "
                "or is an incomplete phrase, rewrite it to be a complete, standalone search query.\n"
                "3. Output ONLY the final question text. No explanations."
            )
        },
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
        do_sample=False, 
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response.strip().replace('"', '')

def query_reformulate(question: str, context: list[str]) -> str:
    """
    Constructs a few-shot prompt to teach the model when to rewrite and when to stay silent.
    """
    history_str = ""
    for i, turn in enumerate(context):
        role = "User" if i % 2 == 0 else "Agent"
        history_str += f"{role}: {turn}\n"
    
    # Few-shot examples to reinforce "No change needed"
    prompt = f"""### Examples:
History:
User: Who directed Inception?
Agent: Christopher Nolan.
Current Question: What is his highest-grossing film?
Standalone Question: What is Christopher Nolan's highest-grossing film?

History:
User: Tell me about the Eiffel Tower.
Agent: It is located in Paris.
Current Question: How tall is the Empire State Building?
Standalone Question: How tall is the Empire State Building? (Note: No change because it is a new standalone topic)

History:
User: How do I bake a chocolate cake?
Agent: You need flour, cocoa, and sugar.
Current Question: How long does it take?
Standalone Question: How long does it take to bake a chocolate cake?

### Task:
History:
{history_str}
Current Question: {question}
Standalone Question:"""

    return generate(prompt)

def main():
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Use a subset for testing if needed
    data = data[:500]

    results = []
    rouge = load("rouge")
    bleu = load("bleu")

    print(f"Starting evaluation on {len(data)} samples...")

    # Clear previous detailed logs
    with open("qrecc_results_detailed.txt", "w") as f:
        f.write("Evaluation Log\n" + "="*20 + "\n")

    for idx, item in enumerate(tqdm(data)):
        context = item.get("Context", [])
        question = item.get("Question", "")
        ground_truth = item.get("Rewrite", "")

        start_time = time.time()
        prediction = query_reformulate(question, context)
        end_time = time.time()
        latency = end_time - start_time
        
        # Determine if the model chose to rewrite or keep original
        status = "REWRITTEN" if prediction.lower() != question.lower() else "KEPT_ORIGINAL"
        
        # Calculate individual sample metrics
        r_score = rouge.compute(predictions=[prediction], references=[ground_truth])
        b_score = bleu.compute(predictions=[prediction], references=[ground_truth])
        
        with open("qrecc_results_detailed.txt", "a") as f:
            f.write(f"Sample ID: {idx} | Status: {status}\n")
            f.write(f"History:\n{''.join(f'- {h}\n' for h in context)}")
            f.write(f"\n")
            f.write(f"Original Query: {question}\n")
            f.write(f"Rewritten Query: {prediction}\n")
            f.write(f"Ground Truth: {ground_truth}\n")
            f.write("\n")
            f.write(f"ROUGE-L: {r_score['rougeL']:.4f}, BLEU: {b_score['bleu']:.4f}\n")
            f.write(f"Latency: {latency:.4f} seconds\n")
            f.write("-" * 30 + "\n")
        
        results.append({
            "original": question,
            "prediction": prediction,
            "reference": ground_truth,
            "status": status,
            "latency": latency
        })

    # Summary Metrics
    preds = [r["prediction"] for r in results]
    refs = [r["reference"] for r in results]
    
    total_rouge = rouge.compute(predictions=preds, references=refs)
    total_bleu = bleu.compute(predictions=preds, references=refs)

    print("\n--- Evaluation Results ---")
    print(f"ROUGE-L: {total_rouge['rougeL']:.4f}")
    print(f"BLEU: {total_bleu['bleu']:.4f}")
    
    # Analyze how often the model rewrites
    rewritten_count = sum(1 for r in results if r["status"] == "REWRITTEN")
    print(f"Rewrite Rate: {rewritten_count/len(results):.2%}")
    
    # Calculate average latency
    avg_latency = sum(r["latency"] for r in results) / len(results)
    print(f"Average Latency: {avg_latency:.4f} seconds")

if __name__ == "__main__":
    main()