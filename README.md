# LLM-DLM Text Summarization

Dự án triển khai mô hình tóm tắt văn bản kết hợp LLM và DLM trên dataset CNN/DailyMail.

---

# Tổng Quan Pipeline

```text
CNN/DailyMail
   ↓
Fine-tune Qwen/LLaMA
   ↓
Sinh draft summary S0
   ↓
Đánh giá baseline S0
   ↓
PP1: Residual DLM refine draft
   ↓
Đánh giá S1
   ↓
PP2: DiffuSeq sinh summary draft
   ↓
Qwen/LLaMA polish
   ↓
Đánh giá PP2
```

---

# 1. Cài Đặt Môi Trường

## Tạo môi trường Python

```bash
python -m venv venv
```

### Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

## Cài thư viện

```bash
pip install torch transformers datasets peft accelerate evaluate rouge_score bert_score tqdm
```

---

# 2. Cài Đặt DiffuSeq

```bash
git clone https://github.com/Shark-NLP/DiffuSeq.git
cd DiffuSeq
pip install -r requirements.txt
cd ..
```

---

# 3. Chuẩn Bị Dataset CNN/DailyMail

## Chạy preprocessing

```bash
python 0_prepare_data.py
```

## Output

```text
data/train.jsonl
data/valid.jsonl
data/test.jsonl
```

## Format dữ liệu

```json
{
  "id": "...",
  "article": "...",
  "reference": "..."
}
```

---

# 🤖 4. Fine-tune LLM Summarizer

# 4.1 Fine-tune Qwen

## Chạy

```bash
python 1_finetune_llm.py
```

## Model mặc định

```python
Qwen/Qwen2.5-1.5B-Instruct
```

## Output model

```text
outputs/qwen_summarizer/final
```

---

# 4.2 Fine-tune LLaMA

## Sửa trong file `1_finetune_llm.py`

```python
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR = "outputs/llama_summarizer"
```

## Chạy

```bash
python 1_finetune_llm.py
```

---

# 5. Sinh Draft Summary S0

# Test 50 mẫu

```bash
python 2_generate_drafts.py \
  --input data/test.jsonl \
  --output data/test_drafts_50.jsonl \
  --limit 50
```

---

# Test 500 mẫu

```bash
python 2_generate_drafts.py \
  --input data/test.jsonl \
  --output data/test_drafts_500.jsonl \
  --limit 500
```

---

# Sinh draft cho train DLM

```bash
python 2_generate_drafts.py \
  --input data/train.jsonl \
  --output data/train_drafts_5000.jsonl \
  --limit 5000
```

```bash
python 2_generate_drafts.py \
  --input data/valid.jsonl \
  --output data/valid_drafts_500.jsonl \
  --limit 500
```

---

# Sinh toàn bộ test set

```bash
python 2_generate_drafts.py \
  --input data/test.jsonl \
  --output data/test_drafts.jsonl \
  --limit -1
```

---

# 6. Đánh Giá Baseline S0

## Sửa trong `4_eval_with_summac.py`

```python
FILE_PATH = "data/test_drafts_500.jsonl"
PRED_KEY = "draft"
```

## Chạy

```bash
python 4_eval_with_summac.py
```

## Metric sử dụng

- ROUGE-1
- ROUGE-2
- ROUGE-L
- ROUGE-Lsum
- BERTScore Precision
- BERTScore Recall
- BERTScore F1
- SUMMAC-ZS

---

# 7. Phương Pháp 1 — Residual DLM Refinement

# Ý tưởng

```text
Article
   ↓
Qwen/LLaMA sinh draft S0
   ↓
Encode latent draft
   ↓
DLM dự đoán Δz
   ↓
z_refined = z_draft + Δz
   ↓
Qwen/LLaMA sinh summary S1
```

---

# 7.1 Train DLM

## Dataset

```python
train_ds = DraftDataset(
    "data/train_drafts_5000.jsonl",
    max_samples=5000
)
```

## Chạy

```bash
python 4_train_dlm.py
```

## Output model

```text
outputs/residual_dlm.pt
```

---

# 7.2 Sinh Summary S1

## Sửa file `5_infer_s1.py`

```python
input_file = "data/test_drafts_500.jsonl"
output_file = "data/test_s1_500.jsonl"
```

## Chạy

```bash
python 5_infer_s1.py
```

---

# 7.3 Đánh Giá S1

## Sửa trong `4_eval_with_summac.py`

```python
FILE_PATH = "data/test_s1_500.jsonl"
PRED_KEY = "s1"
```

## Chạy

```bash
python 4_eval_with_summac.py
```

## So sánh

- S0 = Qwen/LLaMA draft
- S1 = Qwen/LLaMA + DLM refinement

---

# 8. Phương Pháp 2 — DiffuSeq + Qwen/LLaMA Polish

# Ý tưởng

## Train

```text
x = article
y = reference summary

DiffuSeq học p(y | x)
```

## Inference

```text
article
   ↓
DiffuSeq sinh diffusion summary draft
   ↓
Qwen/LLaMA polish
   ↓
Final summary
```

---

# 8.1 Chuẩn Bị Dataset DiffuSeq

```bash
python 8_prepare_diffuseq_cnn.py \
  --input_dir data \
  --output_dir DiffuSeq/datasets/CNNDM \
  --max_train 20000 \
  --max_valid 1000 \
  --max_test 1000
```

## Output

```text
DiffuSeq/datasets/CNNDM/train.jsonl
DiffuSeq/datasets/CNNDM/valid.jsonl
DiffuSeq/datasets/CNNDM/test.jsonl
```

## Format

```json
{
  "src": "article",
  "trg": "reference summary"
}
```

---

# 8.2 Train DiffuSeq

## Chạy

```bash
cd DiffuSeq/scripts
bash 9_train_diffuseq_pp2.sh
cd ../..
```

## Nếu OOM

Giảm các tham số:

```bash
--bsz 32
--microbatch 1
--seq_len 128
--hidden_dim 128
```

## Model output

```text
DiffuSeq/diffusion_models/
```

---

# 8.3 Decode Với DiffuSeq

## Kiểm tra model folder

```bash
ls DiffuSeq/diffusion_models
```

## Sửa trong `10_decode_diffuseq_pp2.sh`

```bash
MODEL_DIR="../diffusion_models/TEN_FOLDER_MODEL"
```

## Chạy

```bash
cd DiffuSeq/scripts
bash 10_decode_diffuseq_pp2.sh
cd ../..
```

## Output

```text
data/diffuseq_test_generations.txt
```

Mỗi dòng là một summary do DiffuSeq sinh ra.

---

# 8.4 Qwen/LLaMA Polish Output

```bash
python 11_polish_diffuseq_qwen.py \
  --test_file data/test.jsonl \
  --diffuseq_file data/diffuseq_test_generations.txt \
  --output data/test_pp2_diffuseq_qwen.jsonl \
  --limit 500
```

## Format output

```json
{
  "article": "...",
  "reference": "...",
  "diffuseq_summary": "...",
  "pp2_diffuseq_qwen": "..."
}
```

---

# 8.5 Đánh Giá PP2

## Sửa trong `4_eval_with_summac.py`

```python
FILE_PATH = "data/test_pp2_diffuseq_qwen.jsonl"
PRED_KEY = "pp2_diffuseq_qwen"
```

## Chạy

```bash
python 4_eval_with_summac.py
```

## Nếu muốn đánh giá DiffuSeq thô

```python
PRED_KEY = "diffuseq_summary"
```

---

# 9. Bảng Kết Quả Cần Báo Cáo

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | SUMMAC-ZS |
|---|---|---|---|---|---|
| S0: Qwen/LLaMA Draft |  |  |  |  |  |
| PP1: Residual DLM Refinement |  |  |  |  |  |
| PP2: DiffuSeq |  |  |  |  |  |
| PP2: DiffuSeq + Qwen/LLaMA Polish |  |  |  |  |  |

---

# 10. Kết Quả Hiện Tại

# Baseline S0

```json
{
  "rouge1": 0.2413,
  "rouge2": 0.1032,
  "rougeL": 0.1747,
  "rougeLsum": 0.2216,
  "bertscore_f1": 0.8659,
  "summac_zs": 0.8569
}
```

---

# PP1 S1

```json
{
  "rouge1": 0.2895,
  "rouge2": 0.1145,
  "rougeL": 0.2095,
  "rougeLsum": 0.2625,
  "bertscore_f1": 0.8711,
  "summac_zs": 0.8171
}
```

---

# Nhận Xét

- PP1 tăng ROUGE và BERTScore.
- SUMMAC giảm.
- DLM refinement giúp summary gần reference hơn.
- Tuy nhiên factual consistency với article có thể giảm.

---

# 11. Thứ Tự Chạy 

# Baseline

```bash
python 0_prepare_data.py
python 1_finetune_llm.py
python 2_generate_drafts.py --input data/test.jsonl --output data/test_drafts_500.jsonl --limit 500
python 4_eval_with_summac.py
```

---

# PP1

```bash
python 2_generate_drafts.py --input data/train.jsonl --output data/train_drafts_5000.jsonl --limit 5000
python 4_train_dlm.py
python 5_infer_s1.py
python 4_eval_with_summac.py
```

---

# PP2 DiffuSeq

```bash
python 8_prepare_diffuseq_cnn.py --input_dir data --output_dir DiffuSeq/datasets/CNNDM --max_train 20000 --max_valid 1000 --max_test 1000

cd DiffuSeq/scripts
bash 9_train_diffuseq_pp2.sh
bash 10_decode_diffuseq_pp2.sh
cd ../..

python 11_polish_diffuseq_qwen.py --test_file data/test.jsonl --diffuseq_file data/diffuseq_test_generations.txt --output data/test_pp2_diffuseq_qwen.jsonl --limit 500

python 4_eval_with_summac.py
```

---

# Tổng Kết

Dự án hiện gồm 3 hướng chính:

1. Baseline LLM summarization.
2. Residual DLM refinement.
3. DiffuSeq + LLM polish.

Mục tiêu cuối cùng:

- So sánh khả năng sinh summary.
- So sánh factual consistency.
- Đánh giá khả năng refinement của DLM.
- Phân tích ưu/nhược điểm giữa LLM và DLM.

