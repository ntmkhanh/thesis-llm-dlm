import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_jsonl(path, max_samples=None):
    data = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data.append(json.loads(line))
    return Dataset.from_list(data)


def make_prompt(article):
    return f"""### Article:
{article}

### Task:
Summarize the article concisely and factually.

### Summary:
"""


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model.config.use_cache = False


lora = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora)
model.print_trainable_parameters()


def preprocess(ex):
    prompt = make_prompt(ex["article"])
    answer = ex["reference"] + tokenizer.eos_token

    full_text = prompt + answer

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=1024,
        padding="max_length"
    )

    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=1024
    )["input_ids"]

    labels = tokenized["input_ids"].copy()
    labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

    tokenized["labels"] = labels
    return tokenized


train_ds = load_jsonl("data/train.jsonl", max_samples=None)
valid_ds = load_jsonl("data/valid.jsonl", max_samples=None)

train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
valid_ds = valid_ds.map(preprocess, remove_columns=valid_ds.column_names)


args = TrainingArguments(
    output_dir="outputs/qwen_summarizer",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds
)

trainer.train()

model.save_pretrained("outputs/qwen_summarizer/final")
tokenizer.save_pretrained("outputs/qwen_summarizer/final")