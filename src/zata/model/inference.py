from __future__ import annotations

import warnings
from pathlib import Path

from zata.model.constants import MAX_SEQ_LENGTH


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    from unsloth import FastLanguageModel


def generate_response(
    finetuned_model: Path,
    prompt: str,
    history: list[dict[str, str]],
) -> str:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(finetuned_model),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    history.append({"role": "user", "content": prompt})

    input_text = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(
            model_inputs.input_ids, generated_ids, strict=False
        )
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
