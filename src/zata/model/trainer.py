from __future__ import annotations

import warnings
from typing import TYPE_CHECKING
from pathlib import Path

from datasets import load_dataset
from transformers import TrainingArguments

from zata.model.utils import formatting_prompts_wrapper_func
from zata.model.constants import MAX_SEQ_LENGTH


if TYPE_CHECKING:
    from zata.model.args import PeftArguments

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    from trl import SFTTrainer
    from unsloth import FastLanguageModel, is_bfloat16_supported


def train(
    model_name: str,
    data_path: Path,
    peft_args: PeftArguments,
    training_args: TrainingArguments,
    save_to_dir: Path,
) -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=peft_args.r,
        target_modules=peft_args.target_modules,
        lora_alpha=peft_args.lora_alpha,
        lora_dropout=peft_args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # prepare dataset
    formatting_prompts_func = formatting_prompts_wrapper_func(tokenizer.eos_token)
    dataset = load_dataset(
        "parquet",
        data_files={"train": str(data_path)},
        split="train",
    )
    dataset.map(formatting_prompts_func, batched=True)

    # trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps=60,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",  # Use this for WandB etc
        ),
    )

    trainer.train()

    model.save_pretrained(save_to_dir)
    tokenizer.save_pretrained(save_to_dir)
