import json

def count_json_samples(file_path):
    count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # bỏ dòng rỗng
            if not line:
                continue

            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError as e:
                print(f"Lỗi JSON ở dòng {count+1}: {e}")

    return count

train_count = count_json_samples("data/train.jsonl")
valid_count = count_json_samples("data/valid.jsonl")
test_count  = count_json_samples("data/test.jsonl")

print(f"Train samples: {train_count}")
print(f"Valid samples: {valid_count}")
print(f"Test samples : {test_count}")
print(f"Total samples: {train_count + valid_count + test_count}")