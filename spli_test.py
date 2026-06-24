import math
from pathlib import Path

input_file = Path(".\data\\valid.jsonl")

output_dir = Path(".\outputs\draft_parts\\valid")
output_dir.mkdir(parents=True, exist_ok=True)

# Đọc dữ liệu
with open(input_file, encoding="utf-8") as f:
    lines = f.readlines()

n = len(lines)
n_parts = 5

chunk_size = math.ceil(n / n_parts)

print(f"Total samples: {n}")
print(f"Chunk size: {chunk_size}")

for i in range(n_parts):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, n)

    out_file = output_dir / f"valid_part_{i+1}.jsonl"

    with open(out_file, "w", encoding="utf-8") as out:
        out.writelines(lines[start:end])

    print(
        f"Part {i+1}: "
        f"{end-start} samples "
        f"({start} -> {end-1})"
    )

print("Done!")