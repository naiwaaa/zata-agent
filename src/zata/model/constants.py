from __future__ import annotations


DEFAULT_MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"

MAX_SEQ_LENGTH = 2048

DATASET_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are impersonating Donald Trump. Answer the question.

### Response:
{}"""

APP_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are impersonating Donald Trump. Answer the question.

### Input:
{}

### Response:
{}"""
