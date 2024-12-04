from __future__ import annotations

import warnings
from pathlib import Path

from zata.model.constants import APP_PROMPT, DEFAULT_MODEL_NAME


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    from unsloth import FastLanguageModel


def generate_response(
    finetuned_model: Path,
    message: str,
    history: list[dict[str, str]],
) -> str:
    history.append({"role": "user", "content": message})

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(DEFAULT_MODEL_NAME),
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    inputs = tokenizer(
        [
            APP_PROMPT.format(
                message,  # input
                "",  # output
            )
        ],
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
    return tokenizer.batch_decode(outputs)
