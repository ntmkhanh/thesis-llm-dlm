import json, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

def load_jsonl(path, max_samples=None):
    data = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data.append(json.loads(line))
    return Dataset.from_list(data)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora)

def make_prompt(article):
    return f"""### Article:
{article}

### Task:
Summarize the article concisely and factually.

### Summary:
"""

def preprocess(ex):
    prompt = make_prompt(ex["article"])
    answer = ex["reference"] + tokenizer.eos_token

    full = prompt + answer

    tokenized = tokenizer(
        full,
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

train_ds = load_jsonl("data/train.jsonl", max_samples=20000).map(preprocess)
valid_ds = load_jsonl("data/valid.jsonl", max_samples=2000).map(preprocess)

args = TrainingArguments(
    output_dir="outputs/llama_summarizer",
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
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds
)

trainer.train()
model.save_pretrained("outputs/llama_summarizer/final")
tokenizer.save_pretrained("outputs/llama_summarizer/final")