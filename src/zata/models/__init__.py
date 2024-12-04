from __future__ import annotations

from transformers import AutoTokenizer, AutoModelForCausalLM


checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"
device = "cuda"  # "cuda" or "cpu"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)


def generate_response(message: str, history: list[dict[str, str]]) -> str:
    history.append({"role": "user", "content": message})
    input_text = tokenizer.apply_chat_template(history, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs, max_new_tokens=100, temperature=0.2, top_p=0.9, do_sample=True
    )
    decoded = tokenizer.decode(outputs[0])
    return str(decoded.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0])
