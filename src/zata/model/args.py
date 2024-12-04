from __future__ import annotations

from unsloth import is_bfloat16_supported
from pydantic import BaseModel


class PromptArguments(BaseModel):
    instruction: str


class PeftArguments(BaseModel):
    r: int = 16
    target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    lora_alpha: int = 16
    lora_dropout: int = 0
    bias: str = "none"
    use_gradient_checkpointing: bool = True
    use_rslora: bool = False


class TrainingArguments(BaseModel):
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4

    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 3
    max_steps: int | None = -1
    warmup_steps: int = 5

    bf16: bool = is_bfloat16_supported()
    fp16: bool = not is_bfloat16_supported()

    optim: str = "adamw_8bit"
    report_to: str = "none"


class FinetuningArguments(BaseModel):
    prompt: PromptArguments
    peft: PeftArguments
    training: TrainingArguments
