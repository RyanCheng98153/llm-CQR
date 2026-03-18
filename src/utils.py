
from pyparsing import Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
SYSTEM_PROMPT_PATH = "prompts/system_prompt.txt"

# Initialize Global Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
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

def generate(prompt: str, max_tokens: int = 128, system_prompt: str = None) -> str:
    if system_prompt is None:
        system_prompt = get_prompt(SYSTEM_PROMPT_PATH)

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
