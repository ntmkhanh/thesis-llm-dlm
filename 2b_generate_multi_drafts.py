import json, argparse, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default="outputs/qwen_summarizer/final")
    p.add_argument("--input", default="data/test.jsonl")
    p.add_argument("--output", default="data/test_multi_drafts_500.jsonl")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--num_return_sequences", type=int, default=4)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--max_input_tokens", type=int, default=900)
    p.add_argument("--max_new_tokens", type=int, default=128)
    return p.parse_args()


def make_prompt(article):
    return f"""### Article:
{article}

### Task:
Summarize the article concisely and factually.

### Summary:
"""


def clean(text):
    if "### Summary:" in text:
        text = text.split("### Summary:")[-1]
    return text.strip()


@torch.no_grad()
def generate_multi(model, tokenizer, article, args):
    inputs = tokenizer(
        make_prompt(article),
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_tokens
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    return [clean(tokenizer.decode(o, skip_special_tokens=True)) for o in outputs]


def main():
    args = parse_args()

    tok = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    n = 0
    with open(args.input, encoding="utf-8") as f, open(args.output, "w", encoding="utf-8") as out:
        for line in tqdm(f, desc="Generating multi drafts"):
            if args.limit > 0 and n >= args.limit:
                break

            ex = json.loads(line)
            ex["drafts"] = generate_multi(model, tok, ex["article"], args)
            out.write(json.dumps(ex, ensure_ascii=False) + "\n")
            out.flush()
            n += 1

    print("Saved:", args.output)


if __name__ == "__main__":
    main()