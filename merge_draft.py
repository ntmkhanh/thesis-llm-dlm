from pathlib import Path

draft_dir = Path("D:\\thesis-llm-dlm\outputs\draft")

parts = sorted(draft_dir.glob("test_drafts_part_*.jsonl"))

merged_file = draft_dir / "test_draft.jsonl"

total = 0

with open(merged_file, "w", encoding="utf-8") as out:
    for file in parts:
        with open(file, encoding="utf-8") as f:
            lines = f.readlines()

        out.writelines(lines)
        total += len(lines)

print(f"Merged {total} samples")