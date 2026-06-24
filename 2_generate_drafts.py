import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", type=str, default="outputs/qwen_summarizer_146/final")

    parser.add_argument("--input", type=str, default="D:\\thesis-llm-dlm\data\\test_parts\\test_part_2.jsonl")
    parser.add_argument("--output", type=str, default="D:\\thesis-llm-dlm\outputs\draft\\test_draft_part_2.jsonl")

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_input_tokens", type=int, default=900)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=4)

    return parser.parse_args()


def make_prompt(article):
    return f"""### Article:
{article}

### Task:
Summarize the article concisely and factually.

### Summary:
"""


def clean_output(text):
    if "### Summary:" in text:
        text = text.split("### Summary:")[-1]
    return text.strip()


@torch.no_grad()
def generate_one(model, tokenizer, article, args):
    prompt = make_prompt(article)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_tokens
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return clean_output(text)


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    n = 0

    with open(args.input, encoding="utf-8") as f, \
         open(args.output, "w", encoding="utf-8") as out:

        for line in tqdm(f, desc="Generating drafts"):
            if args.limit is not None and n >= args.limit:
                break

            ex = json.loads(line)

            draft = generate_one(
                model,
                tokenizer,
                ex["article"],
                args
            )

            ex["draft"] = draft

            out.write(json.dumps(ex, ensure_ascii=False) + "\n")
            out.flush()

            n += 1

    print(f"Saved {n} drafts to {args.output}")


if __name__ == "__main__":
    main()