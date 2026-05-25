# LLM-DLM Text Summarization

Dự án triển khai mô hình tóm tắt văn bản kết hợp LLM và DLM trên dataset CNN/DailyMail.

## Pipeline tổng quát

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
1. Cài đặt môi trường
python -m venv venv
source venv/bin/activate      # Linux
# hoặc
venv\Scripts\activate         # Windows

pip install torch transformers datasets peft accelerate evaluate rouge_score bert_score tqdm
Nếu dùng DiffuSeq:
git clone https://github.com/Shark-NLP/DiffuSeq.git
cd DiffuSeq
pip install -r requirements.txt
cd ..
2. Chuẩn bị dữ liệu CNN/DailyMail
Chạy:
python 0_prepare_data.py
Sau bước này sẽ có:
data/train.jsonl
data/valid.jsonl
data/test.jsonl
Mỗi dòng có dạng:
{
  "id": "...",
  "article": "...",
  "reference": "..."
}
3. Fine-tune Qwen/LLaMA Summarizer
3.1 Fine-tune Qwen
python 1_finetune_llm.py
Mặc định dùng:
Qwen/Qwen2.5-1.5B-Instruct
Kết quả lưu tại:
outputs/qwen_summarizer/final
3.2 Fine-tune LLaMA
Khi có quyền truy cập LLaMA, sửa trong 1_finetune_llm.py:
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR = "outputs/llama_summarizer"
Sau đó chạy:
python 1_finetune_llm.py
4. Sinh draft summary S0
Test trước 50 mẫu:
python 2_generate_drafts.py \
  --input data/test.jsonl \
  --output data/test_drafts_50.jsonl \
  --limit 50
Test 500 mẫu:
python 2_generate_drafts.py \
  --input data/test.jsonl \
  --output data/test_drafts_500.jsonl \
  --limit 500
Sinh draft cho train/valid để train DLM:
python 2_generate_drafts.py \
  --input data/train.jsonl \
  --output data/train_drafts_5000.jsonl \
  --limit 5000

python 2_generate_drafts.py \
  --input data/valid.jsonl \
  --output data/valid_drafts_500.jsonl \
  --limit 500
Nếu muốn chạy toàn bộ test:
python 2_generate_drafts.py \
  --input data/test.jsonl \
  --output data/test_drafts.jsonl \
  --limit -1
5. Đánh giá baseline S0
Mở file 4_eval_with_summac.py, sửa:
FILE_PATH = "data/test_drafts_500.jsonl"
PRED_KEY = "draft"
Chạy:
python 4_eval_with_summac.py
Metric gồm:
ROUGE-1
ROUGE-2
ROUGE-L
ROUGE-Lsum
BERTScore Precision
BERTScore Recall
BERTScore F1
SUMMAC-ZS
6. Phương pháp 1: Residual DLM Refinement
Ý tưởng
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
6.1 Train DLM cho PP1
Trong 4_train_dlm.py, dùng:
train_ds = DraftDataset("data/train_drafts_5000.jsonl", max_samples=5000)
Chạy:
python 4_train_dlm.py
Model lưu tại:
outputs/residual_dlm.pt
6.2 Sinh summary S1
Trong 5_infer_s1.py, sửa input/output:
input_file = "data/test_drafts_500.jsonl"
output_file = "data/test_s1_500.jsonl"
Chạy:
python 5_infer_s1.py
6.3 Đánh giá S1
Trong 4_eval_with_summac.py, sửa:
FILE_PATH = "data/test_s1_500.jsonl"
PRED_KEY = "s1"
Chạy:
python 4_eval_with_summac.py
So sánh:
S0 = Qwen/LLaMA draft
S1 = Qwen/LLaMA + DLM refinement
7. Phương pháp 2: DiffuSeq + Qwen/LLaMA Polish
Ý tưởng
Train:
x = article
y = reference summary
DiffuSeq học p(y | x)

Inference:
article
   ↓
DiffuSeq sinh diffusion summary draft
   ↓
Qwen/LLaMA polish
   ↓
Final summary
7.1 Chuẩn bị dữ liệu cho DiffuSeq
python 8_prepare_diffuseq_cnn.py \
  --input_dir data \
  --output_dir DiffuSeq/datasets/CNNDM \
  --max_train 20000 \
  --max_valid 1000 \
  --max_test 1000
Kết quả:
DiffuSeq/datasets/CNNDM/train.jsonl
DiffuSeq/datasets/CNNDM/valid.jsonl
DiffuSeq/datasets/CNNDM/test.jsonl
Format:
{
  "src": "article",
  "trg": "reference summary"
}
7.2 Train DiffuSeq
Chạy:
cd DiffuSeq/scripts
bash 9_train_diffuseq_pp2.sh
cd ../..
Nếu OOM, giảm trong 9_train_diffuseq_pp2.sh:
--bsz 32
--microbatch 1
--seq_len 128
--hidden_dim 128
Model DiffuSeq lưu trong:
DiffuSeq/diffusion_models/
7.3 Decode bằng DiffuSeq
Xem tên folder model:
ls DiffuSeq/diffusion_models
Sửa trong 10_decode_diffuseq_pp2.sh:
MODEL_DIR="../diffusion_models/TEN_FOLDER_MODEL"
Chạy:
cd DiffuSeq/scripts
bash 10_decode_diffuseq_pp2.sh
cd ../..
Sau đó lấy output generated summary của DiffuSeq và lưu thành:
data/diffuseq_test_generations.txt
Mỗi dòng là một summary do DiffuSeq sinh ra.
7.4 Qwen/LLaMA polish output DiffuSeq
python 11_polish_diffuseq_qwen.py \
  --test_file data/test.jsonl \
  --diffuseq_file data/diffuseq_test_generations.txt \
  --output data/test_pp2_diffuseq_qwen.jsonl \
  --limit 500
Output:
{
  "article": "...",
  "reference": "...",
  "diffuseq_summary": "...",
  "pp2_diffuseq_qwen": "..."
}
7.5 Đánh giá PP2
Trong 4_eval_with_summac.py, sửa:
FILE_PATH = "data/test_pp2_diffuseq_qwen.jsonl"
PRED_KEY = "pp2_diffuseq_qwen"
Chạy:
python 4_eval_with_summac.py
Nếu muốn đánh giá DiffuSeq thô:
PRED_KEY = "diffuseq_summary"
8. Bảng kết quả cần báo cáo
Method	ROUGE-1	ROUGE-2	ROUGE-L	BERTScore F1	SUMMAC-ZS
S0: Qwen/LLaMA Draft					
PP1: Residual DLM Refinement					
PP2: DiffuSeq					
PP2: DiffuSeq + Qwen/LLaMA Polish					
9. Ghi chú quan trọng
Baseline hiện tại
Baseline S0 trên 500 mẫu:
{
  "rouge1": 0.2413,
  "rouge2": 0.1032,
  "rougeL": 0.1747,
  "rougeLsum": 0.2216,
  "bertscore_f1": 0.8659,
  "summac_zs": 0.8569
}
PP1 hiện tại
PP1 S1 trên 500 mẫu:
{
  "rouge1": 0.2895,
  "rouge2": 0.1145,
  "rougeL": 0.2095,
  "rougeLsum": 0.2625,
  "bertscore_f1": 0.8711,
  "summac_zs": 0.8171
}
Nhận xét:
PP1 tăng ROUGE và BERTScore,
nhưng SUMMAC giảm.
Điều này cho thấy DLM refinement giúp summary gần reference hơn nhưng có thể làm giảm factual consistency với article.
10. Thứ tự chạy khuyến nghị
Chạy baseline
python 0_prepare_data.py
python 1_finetune_llm.py
python 2_generate_drafts.py --input data/test.jsonl --output data/test_drafts_500.jsonl --limit 500
python 4_eval_with_summac.py
Chạy PP1
python 2_generate_drafts.py --input data/train.jsonl --output data/train_drafts_5000.jsonl --limit 5000
python 4_train_dlm.py
python 5_infer_s1.py
python 4_eval_with_summac.py
Chạy PP2 DiffuSeq
python 8_prepare_diffuseq_cnn.py --input_dir data --output_dir DiffuSeq/datasets/CNNDM --max_train 20000 --max_valid 1000 --max_test 1000

cd DiffuSeq/scripts
bash 9_train_diffuseq_pp2.sh
bash 10_decode_diffuseq_pp2.sh
cd ../..

python 11_polish_diffuseq_qwen.py --test_file data/test.jsonl --diffuseq_file data/diffuseq_test_generations.txt --output data/test_pp2_diffuseq_qwen.jsonl --limit 500

python 4_eval_with_summac.py