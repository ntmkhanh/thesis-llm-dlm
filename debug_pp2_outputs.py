import json

FILE_PATH = "data/test_pp2_dlm_infer_500.jsonl"

with open(FILE_PATH, encoding="utf-8") as f:
    for i, line in enumerate(f):
        ex = json.loads(line)

        print("=" * 100)
        print("SAMPLE", i)

        print("\nPRED:")
        print(ex["pp2_summary"])

        print("\nREF:")
        print(ex["reference"])

        if i >= 19:
            break