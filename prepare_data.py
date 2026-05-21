from datasets import load_dataset
import json, os

os.makedirs("data", exist_ok=True)

ds = load_dataset("cnn_dailymail", "3.0.0")

def save_split(split, out):
    with open(out, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds[split]):
            item = {
                "id": f"{split}_{i}",
                "article": ex["article"],
                "reference": ex["highlights"]
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

save_split("train", "data/train.jsonl")
save_split("validation", 'data/valid.jsonl')
save_split("test", "data/test.jsonl")