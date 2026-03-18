from typing import List
from src.utils import SYSTEM_PROMPT_PATH, get_prompt, create_prompt, generate

JUDGE_PROMPT_PATH = "prompts/eval_prompt.txt"

def llm_judge(question: str, context: List[str], prediction: str, ground_truth: str) -> float:
    """Uses the LLM to score the rewrite from 0 to 1."""
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
    
    raw_score = generate(prompt=formatted_judge_prompt, max_tokens=10)
    
    try:
        score_match = re.search(r"([0-1]\.\d+|[0-1])", raw_score)
        return float(score_match.group(0)) if score_match else -1.0
    except:
        return -1.0
