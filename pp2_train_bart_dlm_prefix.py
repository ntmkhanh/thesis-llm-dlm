import os, json, torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BartTokenizer, BartModel
from peft import PeftModel

# =====================
# CONFIG
# =====================
BASE_LLM = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_ADAPTER = "outputs/qwen_summarizer/final"

BART_NAME = "facebook/bart-base"
TRAIN_FILE = "data/train.jsonl"
OUT_DIR = "outputs/pp2_bart_dlm"

MAX_ARTICLE_LEN = 512
MAX_SUMMARY_LEN = 128
PREFIX_LEN = 8

BATCH_SIZE = 1
GRAD_ACCUM = 8
EPOCHS = 3
LR = 1e-4

LAMBDA_LATENT = 0.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)


# =====================
# DATA
# =====================
class SummaryDataset(Dataset):
    def __init__(self, path, max_samples=20000):
        self.data = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate(batch):
    return {
        "article": [x["article"] for x in batch],
        "reference": [x["reference"] for x in batch],
    }


# =====================
# MODELS
# =====================
class SemanticDLM(nn.Module):
    """
    PP2: DLM tái tổ chức semantic
    z_article -> z_summary
    """
    def __init__(self, dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, z_article):
        return self.net(z_article)


class PrefixProjector(nn.Module):
    """
    Chuyển z_summary của BART 768-dim
    thành prefix embeddings cho Qwen/LLaMA.
    """
    def __init__(self, in_dim=768, llm_dim=1536, prefix_len=8):
        super().__init__()
        self.prefix_len = prefix_len
        self.llm_dim = llm_dim

        self.proj = nn.Sequential(
            nn.Linear(in_dim, llm_dim),
            nn.Tanh(),
            nn.Linear(llm_dim, prefix_len * llm_dim),
        )

    def forward(self, z):
        p = self.proj(z)
        return p.view(z.size(0), self.prefix_len, self.llm_dim)


# =====================
# LOAD BART ENCODER
# =====================
bart_tok = BartTokenizer.from_pretrained(BART_NAME)
bart = BartModel.from_pretrained(BART_NAME).to(DEVICE)
bart.eval()

for p in bart.parameters():
    p.requires_grad = False


@torch.no_grad()
def encode_bart(texts, max_length):
    inputs = bart_tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(DEVICE)

    outputs = bart.encoder(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
    )

    h = outputs.last_hidden_state
    mask = inputs.attention_mask.unsqueeze(-1)

    z = (h * mask).sum(dim=1) / mask.sum(dim=1)
    return z.float()  # [B, 768]


# =====================
# LOAD LLM
# =====================
llm_tok = AutoTokenizer.from_pretrained(
    LLM_ADAPTER,
    trust_remote_code=True
)

if llm_tok.pad_token is None:
    llm_tok.pad_token = llm_tok.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE_LLM,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

llm = PeftModel.from_pretrained(base, LLM_ADAPTER)
llm.eval()

for p in llm.parameters():
    p.requires_grad = False

LLM_DIM = llm.config.hidden_size

dlm = SemanticDLM(dim=768).to(DEVICE)
projector = PrefixProjector(
    in_dim=768,
    llm_dim=LLM_DIM,
    prefix_len=PREFIX_LEN
).to(DEVICE)

optimizer = torch.optim.AdamW(
    list(dlm.parameters()) + list(projector.parameters()),
    lr=LR
)


# =====================
# PROMPT + LABELS
# =====================
def make_prompt(article):
    return f"""### Article:
{article}

### Task:
Summarize the article concisely and factually.

### Summary:
"""


def build_lm_inputs(article, reference, latent_prefix):
    """
    latent_prefix: [B, PREFIX_LEN, LLM_DIM]
    """
    prompts = [make_prompt(a) for a in article]
    answers = [r + llm_tok.eos_token for r in reference]

    full_texts = [p + a for p, a in zip(prompts, answers)]

    tokenized = llm_tok(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(DEVICE)

    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask

    labels = input_ids.clone()

    # Mask prompt part
    for i, p in enumerate(prompts):
        p_ids = llm_tok(
            p,
            truncation=True,
            max_length=1024,
            padding=False
        )["input_ids"]

        prompt_len = min(len(p_ids), labels.size(1))
        labels[i, :prompt_len] = -100

    token_embeds = llm.get_input_embeddings()(input_ids)

    latent_prefix = latent_prefix.to(token_embeds.dtype)

    inputs_embeds = torch.cat([latent_prefix, token_embeds], dim=1)

    prefix_mask = torch.ones(
        latent_prefix.shape[:2],
        dtype=attention_mask.dtype,
        device=DEVICE
    )

    attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

    prefix_labels = torch.full(
        latent_prefix.shape[:2],
        -100,
        dtype=labels.dtype,
        device=DEVICE
    )

    labels = torch.cat([prefix_labels, labels], dim=1)

    return inputs_embeds, attention_mask, labels


# =====================
# TRAIN
# =====================
dataset = SummaryDataset(TRAIN_FILE, max_samples=20000)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate
)

step = 0

for epoch in range(EPOCHS):
    dlm.train()
    projector.train()

    total_loss = 0

    for batch in tqdm(loader, desc=f"epoch {epoch+1}"):
        # 1. BART encode article/reference
        with torch.no_grad():
            z_article = encode_bart(batch["article"], MAX_ARTICLE_LEN)
            z_ref = encode_bart(batch["reference"], MAX_SUMMARY_LEN)

        # 2. DLM reorganize semantic
        z_summary = dlm(z_article)

        # 3. latent loss
        loss_mse = F.mse_loss(z_summary, z_ref)
        loss_cos = 1 - F.cosine_similarity(z_summary, z_ref, dim=-1).mean()
        loss_latent = loss_mse + 0.1 * loss_cos

        # 4. projector -> prefix
        latent_prefix = projector(z_summary)

        # 5. LLM language loss
        inputs_embeds, attention_mask, labels = build_lm_inputs(
            batch["article"],
            batch["reference"],
            latent_prefix
        )

        outputs = llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False
        )

        loss_lm = outputs.loss

        # 6. total loss
        loss = loss_lm + LAMBDA_LATENT * loss_latent
        loss = loss / GRAD_ACCUM

        loss.backward()

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                list(dlm.parameters()) + list(projector.parameters()),
                1.0
            )
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM
        step += 1

    avg = total_loss / len(loader)
    print(f"epoch {epoch+1} avg_loss = {avg:.4f}")

    torch.save({
        "dlm": dlm.state_dict(),
        "projector": projector.state_dict(),
        "llm_dim": LLM_DIM,
        "prefix_len": PREFIX_LEN,
    }, f"{OUT_DIR}/pp2_epoch{epoch+1}.pt")

torch.save({
    "dlm": dlm.state_dict(),
    "projector": projector.state_dict(),
    "llm_dim": LLM_DIM,
    "prefix_len": PREFIX_LEN,
}, f"{OUT_DIR}/final.pt")

print("Saved:", f"{OUT_DIR}/final.pt")