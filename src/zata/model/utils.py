from __future__ import annotations

from typing import TYPE_CHECKING

from zata.model.constants import DATASET_PROMPT


if TYPE_CHECKING:
    from collections.abc import Callable


def formatting_prompts_wrapper_func(
    eos_token: str,
) -> Callable[[list[str]], dict[str, list[str]]]:
    def func(examples: list[str]) -> dict[str, list[str]]:
        return {
            "text": [DATASET_PROMPT.format(example) + eos_token for example in examples]
        }

    return func
