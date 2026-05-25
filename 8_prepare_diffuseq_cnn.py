import json
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="DiffuSeq/datasets/CNNDM")
    parser.add_argument("--max_train", type=int, default=20000)
    parser.add_argument("--max_valid", type=int, default=1000)
    parser.add_argument("--max_test", type=int, default=1000)
    return parser.parse_args()


def clean_text(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()


def convert_split(in_path, out_path, max_samples=None):
    n = 0

    with open(in_path, encoding="utf-8") as f, \
         open(out_path, "w", encoding="utf-8") as out:

        for line in f:
            if max_samples is not None and max_samples > 0 and n >= max_samples:
                break

            ex = json.loads(line)

            item = {
                "src": clean_text(ex["article"]),
                "trg": clean_text(ex["reference"])
            }

            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1

    print(f"Saved {n} samples to {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    convert_split(
        os.path.join(args.input_dir, "train.jsonl"),
        os.path.join(args.output_dir, "train.jsonl"),
        args.max_train
    )

    convert_split(
        os.path.join(args.input_dir, "valid.jsonl"),
        os.path.join(args.output_dir, "valid.jsonl"),
        args.max_valid
    )

    convert_split(
        os.path.join(args.input_dir, "test.jsonl"),
        os.path.join(args.output_dir, "test.jsonl"),
        args.max_test
    )


if __name__ == "__main__":
    main()