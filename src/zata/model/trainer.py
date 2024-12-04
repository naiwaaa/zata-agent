from __future__ import annotations

import warnings
from typing import TYPE_CHECKING
from pathlib import Path

from datasets import load_dataset
from transformers import TrainingArguments

from zata.model.utils import formatting_prompts_wrapper_func
from zata.model.constants import MAX_SEQ_LENGTH


if TYPE_CHECKING:
    from zata.model.args import FinetuningArguments


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    from trl import SFTTrainer
    from unsloth import FastLanguageModel


def train(
    model_name: str,
    data_path: Path,
    finetuning_args: FinetuningArguments,
    save_to_dir: Path,
) -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        **finetuning_args.peft.model_dump(),
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
            output_dir=str(save_to_dir / "checkpoints"),
            **finetuning_args.training.model_dump(),
        ),
    )

    trainer.train()

    model.save_pretrained(save_to_dir)
    tokenizer.save_pretrained(save_to_dir)
