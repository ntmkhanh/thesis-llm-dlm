# 1a_finetune_llama8b_qlora.py

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# =========================
# CONFIG
# =========================

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

TRAIN_FILE = "data/train.jsonl"
VALID_FILE = "data/valid.jsonl"

OUTPUT_DIR = "outputs/llama8b10_summarizer_qlora"

MAX_LENGTH = 768
MAX_ANSWER_LENGTH = 256

TRAIN_SAMPLES = None
VALID_SAMPLES = None


# =========================
# LOAD DATA
# =========================

def load_jsonl(path, max_samples=None):
    data = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
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


# =========================
# TOKENIZER
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


# =========================
# MODEL 4-BIT QLORA
# =========================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id

model = prepare_model_for_kbit_training(model)


# =========================
# LORA
# =========================

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# =========================
# PREPROCESS - FIX BAD LABELS
# =========================

def preprocess(ex):
    answer = ex["reference"] + tokenizer.eos_token

    answer_tok = tokenizer(
        answer,
        truncation=True,
        max_length=MAX_ANSWER_LENGTH,
        padding=False,
        add_special_tokens=False,
    )

    max_prompt_len = MAX_LENGTH - len(answer_tok["input_ids"])

    if max_prompt_len <= 0:
        max_prompt_len = MAX_LENGTH // 2
        answer_tok["input_ids"] = answer_tok["input_ids"][: MAX_LENGTH - max_prompt_len]

    prompt = make_prompt(ex["article"])

    prompt_tok = tokenizer(
        prompt,
        truncation=True,
        max_length=max_prompt_len,
        padding=False,
        add_special_tokens=True,
    )

    input_ids = prompt_tok["input_ids"] + answer_tok["input_ids"]
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_tok["input_ids"]) + answer_tok["input_ids"]

    input_ids = input_ids[:MAX_LENGTH]
    attention_mask = attention_mask[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]

    pad_len = MAX_LENGTH - len(input_ids)

    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# =========================
# DATASET
# =========================

train_ds = load_jsonl(TRAIN_FILE, TRAIN_SAMPLES)
valid_ds = load_jsonl(VALID_FILE, VALID_SAMPLES)

train_ds = train_ds.map(
    preprocess,
    remove_columns=train_ds.column_names,
)

valid_ds = valid_ds.map(
    preprocess,
    remove_columns=valid_ds.column_names,
)


# =========================
# CHECK BAD LABELS
# =========================

def check_bad_labels(ds, name):
    bad = 0
    for ex in ds:
        if all(x == -100 for x in ex["labels"]):
            bad += 1
    print(f"{name} bad samples:", bad, "/", len(ds))


check_bad_labels(train_ds, "train")
check_bad_labels(valid_ds, "valid")


# =========================
# TRAINING ARGS
# =========================

args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,

    learning_rate=1e-4,
    num_train_epochs=10,
    warmup_ratio=0.03,

    fp16=True,
    bf16=False,

    logging_steps=20,

    eval_strategy="steps",
    eval_steps=500,

    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,

    report_to="none",
    remove_unused_columns=False,

    max_grad_norm=1.0,

    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
)


# =========================
# TRAIN
# =========================

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
)

trainer.train()

model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

print("Saved to:", f"{OUTPUT_DIR}/final")
