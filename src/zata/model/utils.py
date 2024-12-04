from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    from unsloth.chat_templates import get_chat_template


def formatting_prompts_wrapper_func(
    tokenizer: Any,  # noqa: ANN401
    instruction: str,
) -> Callable[[dict[str, list[str]]], dict[str, list[str]]]:
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen2.5",
        system_message=instruction,
    )

    def func(examples: dict[str, list[str]]) -> dict[str, list[str]]:
        return {
            "text": [
                tokenizer.apply_chat_template(
                    [{"from": "assistant", "value": example}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for example in examples["text"]
            ]
        }

    return func
