# README — PP1B: Multi-Draft Latent Refinement

## 1. Tổng quan

PP1B là phần mở rộng của **PP1A — Single-Draft Latent Refinement**.

Ở PP1A, hệ thống chỉ sử dụng một bản tóm tắt nháp duy nhất:

```text
Article → LLM → Single Draft → DLM refine → Final Summary
```

Trong PP1B, thay vì chỉ sinh một draft, LLM sinh nhiều bản nháp khác nhau:

```text
Article → LLM → Multiple Drafts → Selection/Fusion → DLM refine → Final Summary
```

Mục tiêu của PP1B là khai thác nhiều giả thuyết tóm tắt khác nhau do LLM sinh ra, sau đó chọn hoặc kết hợp chúng để tạo ra biểu diễn ngữ nghĩa tốt hơn trước khi DLM tinh chỉnh.

---

## 2. Motivate

Một draft duy nhất có thể mắc lỗi:

- thiếu ý chính;
- diễn đạt chưa tốt;
- sai hoặc thiếu thực thể;
- bỏ sót thông tin quan trọng;
- có hallucination nhẹ.

Khi sinh nhiều draft, mỗi draft có thể chứa một phần thông tin tốt khác nhau. Vì vậy, PP1B khai thác **semantic diversity** từ nhiều draft để cải thiện chất lượng tóm tắt cuối.

Ý tưởng chính:

```text
Nhiều draft = nhiều semantic hypotheses
DLM = refine hoặc tổng hợp các semantic hypotheses đó
```

---

## 3. Kiến trúc tổng quát PP1B

```text
Article
   ↓
Qwen/LLaMA Summarizer
   ↓
Multiple Drafts
[S1, S2, ..., Sk]
   ↓
BART Encoder
   ↓
Latent Drafts
[z1, z2, ..., zk]
   ↓
Best Draft Selection hoặc Latent Fusion
   ↓
z_selected / z_fused
   ↓
Residual DLM Refinement
   ↓
z_refined
   ↓
Latent Projector
   ↓
Qwen/LLaMA Decoder
   ↓
Final Summary
```

---

## 4. Thành phần hệ thống

| Thành phần | Vai trò |
|---|---|
| Qwen/LLaMA | Sinh nhiều draft summary |
| BART Encoder | Mã hoá draft vào latent semantic space |
| Draft Selector | Chọn draft tốt nhất |
| Latent Fusion | Kết hợp latent của nhiều draft |
| Residual DLM | Tinh chỉnh latent đã chọn/fusion |
| Latent Projector | Chuyển latent sang prefix embedding |
| Qwen/LLaMA Decoder | Sinh final summary |

---

## 5. Hai hướng chính trong PP1B

PP1B gồm hai biến thể chính:

```text
PP1B-1: Best Draft Selection
PP1B-2: Multi-Draft Latent Fusion
```

---

# PHẦN A — PP1B-1: Best Draft Selection

## 5.1. Ý tưởng

LLM sinh nhiều draft:

```text
S1, S2, ..., Sk
```

Sau đó hệ thống chọn draft tốt nhất theo một tiêu chí đánh giá.

Ví dụ:

```text
best_draft = argmax(score(Si))
```

Trong đó `score(Si)` có thể là:

- SUMMAC score;
- confidence score;
- reward score;
- semantic similarity score.

Trong implementation hiện tại, sử dụng **SUMMAC-ZS** để chọn draft factual nhất.

---

## 5.2. Pipeline

```text
Article
   ↓
LLM generates k drafts
   ↓
[S1, S2, ..., Sk]
   ↓
SUMMAC Scorer
   ↓
Best Draft
   ↓
DLM refine
   ↓
Final Summary
```

---

## 5.3. Ưu điểm

- Dễ triển khai;
- Ổn định;
- Không cần học thêm fusion model;
- Có thể cải thiện factual consistency nếu dùng SUMMAC để chọn.

---

## 5.4. Nhược điểm

- Chỉ giữ lại một draft;
- Bỏ phí thông tin tốt trong các draft còn lại;
- Không tận dụng hết semantic diversity.

---

# PHẦN B — PP1B-2: Multi-Draft Latent Fusion

## 6.1. Ý tưởng

Thay vì chọn một draft duy nhất, toàn bộ draft được mã hoá vào không gian latent:

```text
S1, S2, ..., Sk
   ↓
BART Encoder
   ↓
z1, z2, ..., zk
```

Sau đó các latent này được kết hợp thành một biểu diễn chung:

```text
z_fused = Fusion(z1, z2, ..., zk)
```

Sau đó DLM tinh chỉnh:

```text
z_refined = z_fused + Δz
```

---

## 6.2. Pipeline

```text
Article
   ↓
LLM generates k drafts
   ↓
[S1, S2, ..., Sk]
   ↓
BART Encoder
   ↓
[z1, z2, ..., zk]
   ↓
Latent Fusion
   ↓
z_fused
   ↓
Residual DLM
   ↓
z_refined
   ↓
Projector
   ↓
Qwen/LLaMA Decode
   ↓
Final Summary
```

---

## 6.3. Ý nghĩa

Mỗi draft là một giả thuyết ngữ nghĩa khác nhau. Latent fusion giúp:

- tổng hợp thông tin;
- giảm phụ thuộc vào một draft duy nhất;
- tìm semantic consensus;
- tăng độ bao phủ thông tin;
- giảm rủi ro hallucination từ một draft cụ thể.

---

# 7. Các chiến lược Latent Fusion

Trong file code hiện tại, PP1B-2 hỗ trợ các fusion key sau:

```text
mean
max
min
first
last
summac_weighted
attention
```

---

## 7.1. Mean Fusion

```text
z_fused = mean(z_i)
```

Công thức:

```text
z_fused = (1/k) * Σ z_i
```

Ý nghĩa:

- lấy trung bình semantic giữa các draft;
- giảm nhiễu;
- đơn giản và ổn định.

Nên chạy đầu tiên vì đây là baseline fusion quan trọng.

---

## 7.2. Max Fusion

```text
z_fused = max(z_i)
```

Lấy giá trị lớn nhất theo từng chiều latent.

Ý nghĩa:

- giữ lại các activation semantic mạnh nhất;
- có thể hữu ích khi một số draft có tín hiệu ngữ nghĩa nổi bật.

Nhược điểm:

- dễ khuếch đại nhiễu;
- không phải lúc nào cũng ổn định.

---

## 7.3. Min Fusion

```text
z_fused = min(z_i)
```

Lấy giá trị nhỏ nhất theo từng chiều latent.

Ý nghĩa:

- fusion bảo thủ hơn;
- chủ yếu dùng làm ablation.

---

## 7.4. First Draft Fusion

```text
z_fused = z_1
```

Dùng draft đầu tiên.

Ý nghĩa:

- baseline gần giống single-draft;
- dùng để kiểm tra multi-draft có thật sự giúp không.

---

## 7.5. Last Draft Fusion

```text
z_fused = z_k
```

Dùng draft cuối cùng.

Ý nghĩa:

- baseline phụ;
- dùng để so sánh với first draft.

---

## 7.6. SUMMAC Weighted Fusion

```text
z_fused = Σ w_i z_i
```

Trong đó:

```text
w_i = softmax(SUMMAC_i)
```

Ý nghĩa:

- draft có SUMMAC cao hơn sẽ có trọng số lớn hơn;
- draft factual hơn ảnh hưởng nhiều hơn vào latent fused.

Pipeline:

```text
Drafts
   ↓
SUMMAC scoring
   ↓
weights
   ↓
weighted latent fusion
```

Ưu điểm:

- có grounding theo article;
- giúp hạn chế hallucination;
- hợp với mục tiêu factual consistency.

Nhược điểm:

- tính SUMMAC chậm hơn;
- phụ thuộc chất lượng NLI scorer.

---

## 7.7. Attention Fusion

Attention Fusion học trọng số cho từng draft:

```text
z1, z2, ..., zk
   ↓
AttentionFusion
   ↓
w1, w2, ..., wk
   ↓
z_fused = Σ w_i z_i
```

Ý tưởng:

- mô hình tự học draft nào quan trọng hơn;
- không cần rule thủ công như mean hoặc SUMMAC.

Lưu ý quan trọng:

```text
Attention Fusion phải được train riêng.
Nếu để random rồi inference thì kết quả không có ý nghĩa.
```

Hiện tại file inference có hỗ trợ key `attention`, nhưng nếu chưa train `AttentionFusion`, nó chỉ là thử nghiệm kỹ thuật.

---

# 8. Residual DLM Refinement

Sau khi có `z_selected` hoặc `z_fused`, DLM không sinh latent mới hoàn toàn.

Thay vào đó DLM học phần hiệu chỉnh:

```text
Δz = DLM(z_fused, z_article)
```

Sau đó:

```text
z_refined = z_fused + Δz
```

Ý nghĩa:

- DLM chỉ sửa nhẹ latent;
- tránh làm hỏng toàn bộ biểu diễn;
- ổn định hơn so với sinh latent từ đầu.

---

# 9. Dữ liệu đầu vào

## 9.1. File gốc

```text
data/train.jsonl
data/valid.jsonl
data/test.jsonl
```

Mỗi dòng:

```json
{
  "id": "...",
  "article": "...",
  "reference": "..."
}
```

---

## 9.2. File multi-draft

Sau khi chạy sinh nhiều draft:

```text
data/test_multi_drafts_500.jsonl
```

Mỗi dòng:

```json
{
  "id": "...",
  "article": "...",
  "reference": "...",
  "drafts": [
    "draft 1",
    "draft 2",
    "draft 3",
    "draft 4"
  ]
}
```

---

## 9.3. File scored drafts

Sau khi chạy SUMMAC selection:

```text
data/test_multi_drafts_500_scored.jsonl
```

Mỗi dòng có thêm:

```json
{
  "best_draft": "...",
  "best_draft_summac": 0.87,
  "draft_scores": [0.82, 0.76, 0.89, 0.80]
}
```

---

# 10. Các file code

| File | Chức năng |
|---|---|
| `2b_generate_multi_drafts.py` | Sinh nhiều draft summary |
| `5b_select_best_draft.py` | Chọn draft tốt nhất bằng SUMMAC |
| `5c_infer_multi_latent_fusion_select.py` | Chạy latent fusion + DLM refine |
| `4_eval_with_summac.py` | Đánh giá ROUGE, BERTScore, SUMMAC |

---

# 11. Cách chạy từng bước

## Bước 1 — Sinh nhiều draft

```bash
python 2b_generate_multi_drafts.py \
  --input data/test.jsonl \
  --output data/test_multi_drafts_500.jsonl \
  --limit 500 \
  --num_return_sequences 4 \
  --num_beams 4
```

Output:

```text
data/test_multi_drafts_500.jsonl
```

---

## Bước 2 — Chọn best draft bằng SUMMAC

```bash
python 5b_select_best_draft.py \
  --input data/test_multi_drafts_500.jsonl \
  --output data/test_best_draft_500.jsonl
```

Output:

```text
data/test_best_draft_500.jsonl
```

Đánh giá best draft bằng:

```python
FILE_PATH = "data/test_best_draft_500.jsonl"
PRED_KEY = "best_draft"
```

---

## Bước 3 — Tạo scored file cho SUMMAC weighted fusion

Nếu muốn chạy `summac_weighted`, cần file có `draft_scores`.

```bash
python 5b_select_best_draft.py \
  --input data/test_multi_drafts_500.jsonl \
  --output data/test_multi_drafts_500_scored.jsonl
```

---

## Bước 4 — Chạy Mean Fusion

```bash
python 5c_infer_multi_latent_fusion_select.py \
  --input data/test_multi_drafts_500.jsonl \
  --output data/test_multi_fusion_mean_500.jsonl \
  --fusion_key mean
```

Đánh giá:

```python
FILE_PATH = "data/test_multi_fusion_mean_500.jsonl"
PRED_KEY = "multi_fusion_mean"
```

---

## Bước 5 — Chạy Max Fusion

```bash
python 5c_infer_multi_latent_fusion_select.py \
  --input data/test_multi_drafts_500.jsonl \
  --output data/test_multi_fusion_max_500.jsonl \
  --fusion_key max
```

Đánh giá:

```python
FILE_PATH = "data/test_multi_fusion_max_500.jsonl"
PRED_KEY = "multi_fusion_max"
```

---

## Bước 6 — Chạy SUMMAC Weighted Fusion

```bash
python 5c_infer_multi_latent_fusion_select.py \
  --input data/test_multi_drafts_500_scored.jsonl \
  --output data/test_multi_fusion_summac_weighted_500.jsonl \
  --fusion_key summac_weighted
```

Đánh giá:

```python
FILE_PATH = "data/test_multi_fusion_summac_weighted_500.jsonl"
PRED_KEY = "multi_fusion_summac_weighted"
```

---

## Bước 7 — Chạy Attention Fusion

Chỉ nên chạy khi đã có hoặc muốn thử module attention.

```bash
python 5c_infer_multi_latent_fusion_select.py \
  --input data/test_multi_drafts_500.jsonl \
  --output data/test_multi_fusion_attention_500.jsonl \
  --fusion_key attention
```

Đánh giá:

```python
FILE_PATH = "data/test_multi_fusion_attention_500.jsonl"
PRED_KEY = "multi_fusion_attention"
```

Lưu ý:

```text
Attention Fusion nếu chưa train thì chỉ là ablation thử nghiệm.
Không nên xem là kết quả chính.
```

---

# 12. Đánh giá

Sử dụng file:

```text
4_eval_with_summac.py
```

Các metric:

| Metric | Ý nghĩa |
|---|---|
| ROUGE-1 | Độ trùng unigram |
| ROUGE-2 | Độ trùng bigram |
| ROUGE-L | Longest common subsequence |
| ROUGE-Lsum | ROUGE-L cho summary |
| BERTScore | Tương đồng ngữ nghĩa |
| SUMMAC-ZS | Factual consistency với article |

---

# 13. Bảng kết quả nên báo cáo

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | SUMMAC-ZS |
|---|---:|---:|---:|---:|---:|
| Single Draft S0 | | | | | |
| PP1A Single-Draft Refinement | | | | | |
| PP1B-1 Best Draft Selection | | | | | |
| PP1B-2 Mean Fusion | | | | | |
| PP1B-2 SUMMAC Weighted Fusion | | | | | |
| PP1B-2 Attention Fusion | | | | | |

---

# 14. Cách diễn giải kết quả

## Trường hợp 1 — ROUGE tăng, SUMMAC giảm

Điều này nghĩa là:

```text
summary giống reference hơn nhưng factual consistency giảm
```

Có thể diễn giải là:

```text
DLM giúp cải thiện reference alignment nhưng có nguy cơ semantic drift.
```

---

## Trường hợp 2 — SUMMAC tăng, ROUGE giảm nhẹ

Điều này nghĩa là:

```text
summary trung thành với article hơn nhưng ít giống reference hơn
```

Có thể diễn giải là:

```text
Fusion giúp tăng factual consistency nhưng chưa tối ưu lexical overlap.
```

---

## Trường hợp 3 — Cả ROUGE và SUMMAC đều tăng

Đây là kết quả tốt nhất.

Có thể kết luận:

```text
Multi-draft latent fusion giúp cải thiện cả chất lượng tóm tắt và tính nhất quán thông tin.
```

---

# 15. Insight nghiên cứu

PP1B có ý nghĩa vì nó cho phép mô hình:

- không phụ thuộc vào một draft duy nhất;
- tận dụng nhiều giả thuyết tóm tắt;
- kết hợp semantic information trong latent space;
- dùng DLM để refine semantic representation sau fusion.

Best Draft Selection thiên về:

```text
reranking
```

Multi-Draft Latent Fusion thiên về:

```text
semantic ensemble + latent refinement
```

---

# 16. Kết luận

PP1B là mở rộng quan trọng của PP1A.

Nó gồm hai nhánh:

```text
PP1B-1: chọn draft tốt nhất
PP1B-2: fusion latent của nhiều draft
```

Trong đó:

- PP1B-1 đơn giản, ổn định, dễ giải thích;
- PP1B-2 có novelty cao hơn, tận dụng semantic diversity tốt hơn;
- SUMMAC Weighted Fusion là biến thể đáng ưu tiên vì có yếu tố factual consistency;
- Attention Fusion chỉ nên dùng sau khi train riêng attention module.

