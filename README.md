# Latent Diffusion for Text Summarization

## Ý tưởng chính

Trong code này, mô hình được thiết kế theo hướng kết hợp **Latent Diffusion Model (DLM)** với mô hình sinh văn bản như **Qwen/LLaMA** để thực hiện bài toán tóm tắt văn bản.

Pipeline tổng quát:

```text
Article
  ↓
Article Encoder
  ↓
z_article

Reference Summary
  ↓
Summary Encoder
  ↓
z_ref
```

Trong đó:

- `z_article` là latent embedding biểu diễn nội dung bài báo.
- `z_ref` là latent embedding biểu diễn bản tóm tắt tham chiếu.
- DLM học cách biến latent nhiễu trở lại latent summary sạch.

---

## 1. Forward Diffusion

Ở bước forward diffusion, ta thêm nhiễu Gaussian vào latent summary tham chiếu `z_ref`.

Công thức:

```text
z_t = sqrt(alpha_bar_t) * z_ref + sqrt(1 - alpha_bar_t) * eps
```

Trong đó:

- `z_ref`: latent embedding của reference summary.
- `eps`: nhiễu Gaussian.
- `t`: timestep diffusion.
- `alpha_bar_t`: hệ số điều khiển mức độ nhiễu tại timestep `t`.
- `z_t`: latent summary đã bị thêm nhiễu.

Ý nghĩa:

```text
z_ref → z_1 → z_2 → ... → z_T
```

Càng về sau, latent càng bị nhiễu nhiều hơn.

---

## 2. Denoiser học dự đoán nhiễu

Denoiser nhận vào:

```text
DLM(z_t, t, z_article)
```

và học dự đoán nhiễu `eps`.

Công thức học:

```text
DLM(z_t, t, z_article) → eps
```

Trong đó:

- `z_t`: latent summary đang bị nhiễu.
- `t`: timestep hiện tại.
- `z_article`: latent biểu diễn nội dung article.
- `eps`: nhiễu thật đã được thêm vào trong forward diffusion.

Mục tiêu huấn luyện:

```text
pred_eps ≈ eps
```

Loss thường dùng:

```text
Loss = MSE(pred_eps, eps)
```

Ý nghĩa:

DLM không sinh text trực tiếp.  
DLM học cách khử nhiễu trong latent space, có điều kiện theo nội dung bài báo.

---

## 3. Reverse Diffusion

Sau khi huấn luyện, quá trình sinh latent summary bắt đầu từ nhiễu ngẫu nhiên `z_T`.

```text
z_T → z_{T-1} → ... → z_0
```

Ở mỗi bước, DLM dự đoán phần nhiễu cần loại bỏ:

```text
pred_eps = DLM(z_t, t, z_article)
```

Sau đó cập nhật dần từ latent nhiễu về latent sạch.

Kết quả cuối cùng:

```text
z_0 = z_summary
```

Trong đó:

- `z_0` là latent summary sau khi reverse diffusion hoàn tất.
- `z_summary` là biểu diễn latent của bản tóm tắt cần sinh.

---

## 4. Sinh văn bản từ latent summary

Sau khi thu được `z_summary`, ta dùng nó như một dạng **latent prefix** cho mô hình sinh văn bản.

```text
z_summary
  ↓
Latent Prefix Projector
  ↓
Qwen/LLaMA Decoder
  ↓
Generated Summary Text
```

Ý tưởng:

```text
z_0 = z_summary
rồi Qwen/LLaMA sinh text từ latent prefix
```

Qwen/LLaMA không tự tạo summary từ đầu, mà nhận tín hiệu ngữ nghĩa đã được DLM tổ chức lại trong latent space.

---

## 5. Tóm tắt toàn bộ pipeline

```text
Training phase:

Article → Encoder → z_article
Reference Summary → Encoder → z_ref

z_ref + noise → z_t

DLM(z_t, t, z_article) → pred_eps

Loss = MSE(pred_eps, eps)
```

```text
Inference phase:

Article → Encoder → z_article

Random noise → z_T

Reverse diffusion:
z_T → z_{T-1} → ... → z_0

z_0 = z_summary

Qwen/LLaMA(z_summary as latent prefix) → summary text
```

---

## 6. Vai trò của từng thành phần

| Thành phần | Vai trò |
|---|---|
| Article Encoder | Mã hóa bài báo thành `z_article` |
| Summary Encoder | Mã hóa reference summary thành `z_ref` |
| Forward Diffusion | Thêm nhiễu vào `z_ref` để tạo `z_t` |
| DLM / Denoiser | Học khử nhiễu latent summary dựa trên article |
| Reverse Diffusion | Tạo lại latent summary sạch từ nhiễu |
| Qwen/LLaMA | Sinh văn bản summary từ latent prefix |

---

## 7. Kết luận

Mô hình này không dùng diffusion để sinh token trực tiếp.

Thay vào đó, diffusion hoạt động trong **latent space**:

```text
Noisy summary latent → Clean summary latent
```

Sau đó, Qwen/LLaMA đảm nhiệm bước cuối:

```text
Clean summary latent → Summary text
```

Vì vậy, DLM đóng vai trò tổ chức và tinh chỉnh biểu diễn ngữ nghĩa của summary trước khi mô hình ngôn ngữ sinh ra văn bản cuối cùng.
