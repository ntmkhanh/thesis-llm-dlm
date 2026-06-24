import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model


# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

TRAIN_FILE = "data/train.jsonl"
VALID_FILE = "data/valid.jsonl"

OUTPUT_DIR = "outputs/llama_summarizer"

MAX_LENGTH = 1024
MAX_ANSWER_LENGTH = 160
MAX_PROMPT_LENGTH = MAX_LENGTH - MAX_ANSWER_LENGTH

# None = train full CNN/DailyMail
TRAIN_SAMPLES = None
VALID_SAMPLES = None
=======
TRAIN_SAMPLES = 100000
VALID_SAMPLES = 10000


# =========================================================
# DATA LOADING
# =========================================================

def load_jsonl(path, max_samples=None):
    data = []

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break

            ex = json.loads(line)

            if "article" not in ex or "reference" not in ex:
                continue

            if not ex["article"].strip() or not ex["reference"].strip():
                continue

            data.append(ex)

    return Dataset.from_list(data)


def make_prompt(article):
    article = article.strip()

    return f"""### Article:
{article}

### Task:
Summarize the article concisely and factually.

### Summary:
"""


# =========================================================
# TOKENIZER + MODEL
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False
model.gradient_checkpointing_enable()


# =========================================================
# LORA
# =========================================================

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


# =========================================================
# PREPROCESS
# =========================================================

def preprocess(ex):
    prompt = make_prompt(ex["article"])
    answer = ex["reference"].strip() + tokenizer.eos_token

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

    input_ids = input_ids[:MAX_LENGTH]
    attention_mask = attention_mask[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]

    pad_len = MAX_LENGTH - len(input_ids)

    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len

    if all(x == -100 for x in labels):
        labels[-1] = tokenizer.eos_token_id

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# =========================================================
# DATASET
# =========================================================

print("Loading dataset...")

train_ds = load_jsonl(TRAIN_FILE, TRAIN_SAMPLES)
valid_ds = load_jsonl(VALID_FILE, VALID_SAMPLES)

print("Train samples:", len(train_ds))
print("Valid samples:", len(valid_ds))

train_ds = train_ds.map(
    preprocess,
    remove_columns=train_ds.column_names,
    desc="Tokenizing train",
)

valid_ds = valid_ds.map(
    preprocess,
    remove_columns=valid_ds.column_names,
    desc="Tokenizing valid",
)


# =========================================================
# CHECK DATA
# =========================================================

def check_bad_labels(ds, name):
    bad = 0

    for ex in ds:
        if all(x == -100 for x in ex["labels"]):
            bad += 1

    print(f"{name} bad label samples: {bad} / {len(ds)}")


def check_token_stats(ds, name):
    total_lens = []
    label_lens = []

    for ex in ds:
        total_lens.append(sum(ex["attention_mask"]))
        label_lens.append(sum(1 for x in ex["labels"] if x != -100))

    print(f"\n{name} token stats")
    print("avg total tokens:", sum(total_lens) / len(total_lens))
    print("max total tokens:", max(total_lens))
    print("avg label tokens:", sum(label_lens) / len(label_lens))
    print("min label tokens:", min(label_lens))
    print("max label tokens:", max(label_lens))


check_bad_labels(train_ds, "train")
check_bad_labels(valid_ds, "valid")

check_token_stats(train_ds, "train")
check_token_stats(valid_ds, "valid")


# =========================================================
# TRAINING ARGS
# =========================================================

args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,

    learning_rate=2e-5,
    num_train_epochs=10,

    fp16=True,
    bf16=False,

    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    max_grad_norm=0.5,

    logging_steps=50,

    eval_strategy="steps",
    eval_steps=2000,

    save_strategy="steps",
    save_steps=2000,
    save_total_limit=2,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    report_to="none",
    remove_unused_columns=False,
<<<<<<< HEAD

    dataloader_num_workers=2,
=======
    # max_grad_norm=1.0,
    max_grad_norm=0.5,
>>>>>>> 5fcdd09 (back)
)


# =========================================================
# TRAIN
# =========================================================

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2,
            early_stopping_threshold=0.0005,
        )
    ],
)

trainer.train()


# =========================================================
# SAVE FINAL
# =========================================================

final_dir = f"{OUTPUT_DIR}/final"

model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

print("Saved to:", final_dir)
