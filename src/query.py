from typing import List
from src.utils import REWRITE_PROMPT_PATH, get_prompt, create_prompt, generate

REWRITE_PROMPT_PATH = "prompts/rewrite_prompt.txt"

def query_reformulate(question: str, context: List[str]) -> str:
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
    return generate(prompt=formatted_user_prompt)
