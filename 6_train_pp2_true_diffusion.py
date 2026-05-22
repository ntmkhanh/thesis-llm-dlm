import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BartTokenizer,
    BartModel,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel


# =========================================================
# CONFIG
# =========================================================

BART_NAME = "facebook/bart-base"

BASE_LLM = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_ADAPTER = "outputs/qwen_summarizer/final"

TRAIN_FILE = "data/train.jsonl"
OUT_DIR = "outputs/pp2_true_diffusion"

MAX_ARTICLE_LEN = 512
MAX_SUMMARY_LEN = 128

BATCH_SIZE = 1
GRAD_ACCUM = 8
EPOCHS = 3
LR = 1e-4

T = 1000
PREFIX_LEN = 8
LAMBDA_LM = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================
# DATASET
# =========================================================

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
        ex = self.data[idx]
        return {
            "article": ex["article"],
            "reference": ex["reference"],
        }


def collate_fn(batch):
    return {
        "article": [x["article"] for x in batch],
        "reference": [x["reference"] for x in batch],
    }


# =========================================================
# DIFFUSION SCHEDULE
# =========================================================

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule giống DDPM cải tiến.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)

    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * math.pi * 0.5
    ) ** 2

    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.0001, 0.9999)

    return betas


class GaussianDiffusion:
    def __init__(self, timesteps=1000, device="cuda"):
        self.timesteps = timesteps
        self.device = device

        betas = cosine_beta_schedule(timesteps).to(device)

        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_bar = alphas_bar

        self.sqrt_alphas_bar = torch.sqrt(alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    def q_sample(self, z0, t, noise=None):
        """
        Forward diffusion:
        z_t = sqrt(alpha_bar_t) z0 + sqrt(1-alpha_bar_t) eps
        """
        if noise is None:
            noise = torch.randn_like(z0)

        sqrt_ab = self.sqrt_alphas_bar[t].view(-1, 1)
        sqrt_omab = self.sqrt_one_minus_alphas_bar[t].view(-1, 1)

        z_t = sqrt_ab * z0 + sqrt_omab * noise
        return z_t, noise

    @torch.no_grad()
    def p_sample(self, denoiser, z_t, t, z_article):
        """
        Reverse step:
        z_t -> z_{t-1}
        Denoiser dự đoán epsilon.
        """
        batch_size = z_t.size(0)
        t_batch = torch.full(
            (batch_size,),
            t,
            device=z_t.device,
            dtype=torch.long
        )

        eps_pred = denoiser(z_t, t_batch, z_article)

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_bar[t]

        coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)

        mean = (1.0 / torch.sqrt(alpha_t)) * (z_t - coef * eps_pred)

        if t == 0:
            return mean

        noise = torch.randn_like(z_t)
        sigma = torch.sqrt(beta_t)

        return mean + sigma * noise

    @torch.no_grad()
    def sample(self, denoiser, z_article, shape):
        """
        Start from pure noise z_T.
        """
        z = torch.randn(shape, device=self.device)

        for t in reversed(range(self.timesteps)):
            z = self.p_sample(
                denoiser=denoiser,
                z_t=z,
                t=t,
                z_article=z_article
            )

        return z


# =========================================================
# MODELS
# =========================================================

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        """
        Sinusoidal timestep embedding.
        t: [B]
        """
        half = self.dim // 2
        device = t.device

        freqs = torch.exp(
            -math.log(10000) *
            torch.arange(0, half, device=device).float() / half
        )

        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)

        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return self.mlp(emb)


class ConditionalDenoiser(nn.Module):
    """
    DLM thật theo kiểu conditional latent DDPM.

    Input:
        z_t       : noisy summary latent
        t         : timestep
        z_article : condition from BART encoder

    Output:
        eps_pred  : predicted Gaussian noise
    """
    def __init__(self, dim=768):
        super().__init__()

        self.time_emb = TimeEmbedding(dim)

        self.net = nn.Sequential(
            nn.Linear(dim * 3, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(dim * 4, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(dim * 4, dim),
        )

    def forward(self, z_t, t, z_article):
        t_emb = self.time_emb(t)

        x = torch.cat(
            [z_t, z_article, t_emb],
            dim=-1
        )

        eps_pred = self.net(x)
        return eps_pred


class PrefixProjector(nn.Module):
    """
    Project z_summary từ BART latent dim 768
    sang prefix embeddings của Qwen/LLaMA.
    """
    def __init__(self, in_dim=768, llm_dim=1536, prefix_len=8):
        super().__init__()

        self.prefix_len = prefix_len
        self.llm_dim = llm_dim

        self.proj = nn.Sequential(
            nn.Linear(in_dim, llm_dim),
            nn.Tanh(),
            nn.Linear(llm_dim, prefix_len * llm_dim)
        )

    def forward(self, z_summary):
        prefix = self.proj(z_summary)
        prefix = prefix.view(
            z_summary.size(0),
            self.prefix_len,
            self.llm_dim
        )
        return prefix


# =========================================================
# LOAD BART ENCODER
# =========================================================

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
        max_length=max_length
    ).to(DEVICE)

    outputs = bart.encoder(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask
    )

    h = outputs.last_hidden_state
    mask = inputs.attention_mask.unsqueeze(-1)

    z = (h * mask).sum(dim=1) / mask.sum(dim=1)

    return z.float()


# =========================================================
# LOAD LLM DECODER
# =========================================================

llm_tok = AutoTokenizer.from_pretrained(
    LLM_ADAPTER,
    trust_remote_code=True
)

if llm_tok.pad_token is None:
    llm_tok.pad_token = llm_tok.eos_token

base_llm = AutoModelForCausalLM.from_pretrained(
    BASE_LLM,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

llm = PeftModel.from_pretrained(base_llm, LLM_ADAPTER)
llm.eval()

for p in llm.parameters():
    p.requires_grad = False

LLM_DIM = llm.config.hidden_size


# =========================================================
# PROMPT + LM LOSS
# =========================================================

def make_prompt(article):
    return f"""### Article:
{article}

### Task:
Summarize the article concisely and factually.

### Summary:
"""


def build_lm_inputs(article_list, reference_list, latent_prefix):
    prompts = [make_prompt(a) for a in article_list]
    answers = [r + llm_tok.eos_token for r in reference_list]

    full_texts = [p + a for p, a in zip(prompts, answers)]

    tokenized = llm_tok(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(DEVICE)

    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask

    labels = input_ids.clone()

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

    inputs_embeds = torch.cat(
        [latent_prefix, token_embeds],
        dim=1
    )

    prefix_mask = torch.ones(
        latent_prefix.shape[:2],
        dtype=attention_mask.dtype,
        device=DEVICE
    )

    attention_mask = torch.cat(
        [prefix_mask, attention_mask],
        dim=1
    )

    prefix_labels = torch.full(
        latent_prefix.shape[:2],
        -100,
        dtype=labels.dtype,
        device=DEVICE
    )

    labels = torch.cat(
        [prefix_labels, labels],
        dim=1
    )

    return inputs_embeds, attention_mask, labels


# =========================================================
# INIT
# =========================================================

diffusion = GaussianDiffusion(
    timesteps=T,
    device=DEVICE
)

denoiser = ConditionalDenoiser(dim=768).to(DEVICE)

projector = PrefixProjector(
    in_dim=768,
    llm_dim=LLM_DIM,
    prefix_len=PREFIX_LEN
).to(DEVICE)

optimizer = torch.optim.AdamW(
    list(denoiser.parameters()) + list(projector.parameters()),
    lr=LR
)

dataset = SummaryDataset(TRAIN_FILE, max_samples=20000)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)


# =========================================================
# TRAIN
# =========================================================

global_step = 0

for epoch in range(EPOCHS):
    denoiser.train()
    projector.train()

    total_loss = 0.0

    for batch in tqdm(loader, desc=f"epoch {epoch + 1}"):

        with torch.no_grad():
            z_article = encode_bart(
                batch["article"],
                MAX_ARTICLE_LEN
            )

            z_ref = encode_bart(
                batch["reference"],
                MAX_SUMMARY_LEN
            )

        bsz = z_ref.size(0)

        t = torch.randint(
            low=0,
            high=T,
            size=(bsz,),
            device=DEVICE,
            dtype=torch.long
        )

        z_t, noise = diffusion.q_sample(
            z0=z_ref,
            t=t
        )

        eps_pred = denoiser(
            z_t=z_t,
            t=t,
            z_article=z_article
        )

        loss_diff = F.mse_loss(eps_pred, noise)

        # Train projector bằng clean z_ref để LLM học dùng prefix đúng.
        latent_prefix = projector(z_ref)

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

        loss = loss_diff + LAMBDA_LM * loss_lm
        loss = loss / GRAD_ACCUM

        loss.backward()

        if (global_step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                list(denoiser.parameters()) + list(projector.parameters()),
                1.0
            )

            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM
        global_step += 1

    avg_loss = total_loss / len(loader)

    print(f"epoch={epoch + 1} avg_loss={avg_loss:.4f}")

    ckpt = {
        "denoiser": denoiser.state_dict(),
        "projector": projector.state_dict(),
        "llm_dim": LLM_DIM,
        "prefix_len": PREFIX_LEN,
        "T": T,
    }

    torch.save(
        ckpt,
        f"{OUT_DIR}/epoch_{epoch + 1}.pt"
    )

torch.save(
    {
        "denoiser": denoiser.state_dict(),
        "projector": projector.state_dict(),
        "llm_dim": LLM_DIM,
        "prefix_len": PREFIX_LEN,
        "T": T,
    },
    f"{OUT_DIR}/final.pt"
)

print("Saved:", f"{OUT_DIR}/final.pt")

# BART encoder: frozen
# Qwen/LLaMA: frozen
# DLM denoiser: train
# Projector: train