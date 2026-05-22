import json, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER = "outputs/llama_summarizer/final"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

def prompt(article):
    return f"""### Article:
{article}

### Task:
Summarize the article concisely and factually.

### Summary:
"""

@torch.no_grad()
def generate(article):
    inputs = tokenizer(
        prompt(article),
        return_tensors="pt",
        truncation=True,
        max_length=900
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=4,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text.split("### Summary:")[-1].strip()

def run(in_path, out_path):
    with open(in_path, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
        for line in tqdm(f):
            ex = json.loads(line)
            draft = generate(ex["article"])
            ex["draft"] = draft
            out.write(json.dumps(ex, ensure_ascii=False) + "\n")

run("data/train.jsonl", "data/train_drafts.jsonl")
run("data/valid.jsonl", "data/valid_drafts.jsonl")
run("data/test.jsonl", "data/test_drafts.jsonl")