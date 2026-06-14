import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


# =========================
# CONFIG
# =========================

# MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

TRAIN_FILE = "data/train.jsonl"
VALID_FILE = "data/valid.jsonl"

# OUTPUT_DIR = "outputs/qwen_summarizer"

OUTPUT_DIR = "outputs/llama_summarizer"

MAX_LENGTH = 1024
MAX_ANSWER_LENGTH = 160
MAX_PROMPT_LENGTH = MAX_LENGTH - MAX_ANSWER_LENGTH

TRAIN_SAMPLES = 50000
VALID_SAMPLES = 5000


# =========================
# LOAD DATA
# =========================

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


# =========================
# TOKENIZER + MODEL
# =========================

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
# PREPROCESS — FIX NAN LOSS
# =========================

def preprocess(ex):
    prompt = make_prompt(ex["article"])
    answer = ex["reference"] + tokenizer.eos_token

    prompt_tok = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_PROMPT_LENGTH,
        padding=False,
    )

    answer_tok = tokenizer(
        answer,
        truncation=True,
        max_length=MAX_ANSWER_LENGTH,
        padding=False,
    )

    input_ids = prompt_tok["input_ids"] + answer_tok["input_ids"]
    attention_mask = [1] * len(input_ids)

    labels = [-100] * len(prompt_tok["input_ids"]) + answer_tok["input_ids"]

    if len(input_ids) > MAX_LENGTH:
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

train_ds = load_jsonl(TRAIN_FILE, max_samples=TRAIN_SAMPLES)
valid_ds = load_jsonl(VALID_FILE, max_samples=VALID_SAMPLES)

train_ds = train_ds.map(
    preprocess,
    remove_columns=train_ds.column_names,
)

valid_ds = valid_ds.map(
    preprocess,
    remove_columns=valid_ds.column_names,
)


# =========================
# CHECK LABEL BUG
# =========================

def check_bad_labels(ds, name):
    bad = 0
    for ex in ds:
        if all(x == -100 for x in ex["labels"]):
            bad += 1
    print(f"{name} bad samples:", bad, "/", len(ds))


check_bad_labels(train_ds, "train")
check_bad_labels(valid_ds, "valid")


def check_token_stats(ds, name):
    total_lens = []
    label_lens = []

    for ex in ds:
        total_lens.append(sum(ex["attention_mask"]))
        label_lens.append(sum(1 for x in ex["labels"] if x != -100))

    print(f"\n{name}")
    print("avg total tokens :", sum(total_lens) / len(total_lens))
    print("max total tokens :", max(total_lens))
    print("avg label tokens :", sum(label_lens) / len(label_lens))
    print("min label tokens :", min(label_lens))
    print("max label tokens :", max(label_lens))


check_token_stats(train_ds, "train")
check_token_stats(valid_ds, "valid")


# =========================
# TRAINING ARGS
# =========================

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    # learning_rate=1e-4,
    num_train_epochs=10,

    fp16=True,
    # Nếu vẫn nan thì đổi:
    # fp16=False,
    # bf16=True,  # chỉ dùng nếu GPU hỗ trợ bf16

    logging_steps=20,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,

    report_to="none",
    remove_unused_columns=False,
    # max_grad_norm=1.0,
    max_grad_norm=0.5,
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
