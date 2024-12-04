from __future__ import annotations

from dataclasses import dataclass

import transformers
from unsloth import is_bfloat16_supported


class PeftArguments:
    r = 16
    lora_alpha = 16
    lora_dropout = 0
    target_modules = (
        [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    max_seq_length = 2048

    per_device_train_batch_size = 2
    gradient_accumulation_steps = 4
    warmup_steps = 5
    max_steps = 60
    learning_rate = 2e-4
    fp16 = not is_bfloat16_supported()
    bf16 = is_bfloat16_supported()
    logging_steps = 1
    optim = "adamw_8bit"
    weight_decay = 0.01
    lr_scheduler_type = "linear"
    seed = 3407
    output_dir = "outputs"
    report_to = "none"  # Use this for WandB etc
