from __future__ import annotations

import warnings
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    from zata.model.args import FinetuningArguments


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    from unsloth import FastLanguageModel


def generate_response_wrapper(
    finetuned_model: str,
    finetuning_args: FinetuningArguments,
) -> Callable[[str, list[dict[str, str]]], str]:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=finetuned_model,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    def func(prompt: str, history: list[dict[str, str]]) -> str:
        history.append({
            "role": "user",
            "content": f"{finetuning_args.prompt.instruction} {prompt}",
        })

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

        return str(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])

    return func
