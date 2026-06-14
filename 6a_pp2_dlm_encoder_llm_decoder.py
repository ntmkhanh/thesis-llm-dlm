import os
import re
import json
import math
import argparse
import random
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


"""
PP2: DLM Latent Planning + Latent Projector + LLM Decoder
---------------------------------------------------------

Flow:
    Article
      ↓
    DLM
      ↓
    Latent Plan z
      ↓
    Latent Projector
      ↓
    LLM Decoder
      ↓
    Summary

Modes:
    1) train_dlm
       Train DLM to denoise latent summary plan conditioned on article.

    2) train_projector
       Freeze DLM + LLM. Train Latent Projector so LLM can consume latent plan
       as soft prefix.

    3) oracle_infer
       Use gold latent plan from reference. This checks projector + LLM upper bound.

    4) infer
       Article -> DLM sampled latent plan -> projector -> LLM summary.
"""


BASE_LLM = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_ADAPTER = "outputs/qwen_summarizer/final"

TRAIN_FILE = "data/train.jsonl"
TEST_FILE = "data/test.jsonl"

OUT_DIR = "outputs/pp2_latent_plan_projector"

MAX_ARTICLE_LEN = 512
MAX_PLAN_LEN = 96
LLM_MAX_LEN = 1536

BATCH_SIZE = 1
GRAD_ACCUM = 8
EPOCHS = 10

LR_DLM = 1e-5
LR_PROJECTOR = 1e-6

T = 2000
SAMPLE_STEPS = 500

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 102):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def clean_summary(text: str, max_sentences: int = 3, max_words: int = 90) -> str:
    if text is None:
        return ""

    text = text.strip()
    text = re.sub(r"(?:-->\s*)+", " ", text)
    text = re.sub(r"<\|.*?\|>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if "### Summary:" in text:
        text = text.split("### Summary:")[-1].strip()

    text = re.sub(r"^###\s*Summary\s*: ?", "", text, flags=re.I).strip()
    text = re.sub(r"^Summary\s*: ?", "", text, flags=re.I).strip()
    text = re.sub(r"^[-*]\s+", "", text).strip()

    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in sents if s.strip()]
    if len(sents) > max_sentences:
        text = " ".join(sents[:max_sentences])

    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
        if not text.endswith((".", "!", "?")):
            text += "."

    return text.strip()


class SummaryDataset(Dataset):
    def __init__(self, path: str, max_samples: Optional[int] = None):
        self.data = []

        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break

                ex = json.loads(line)
                article = ex.get("article", "").strip()
                reference = (
                    ex.get("reference")
                    or ex.get("highlights")
                    or ex.get("summary")
                    or ""
                ).strip()

                if not article or not reference:
                    continue

                self.data.append({
                    "id": ex.get("id", str(i)),
                    "article": article,
                    "reference": reference,
                })

        print(f"Loaded {len(self.data)} samples from {path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch: List[Dict[str, Any]]):
    return {
        "id": [x["id"] for x in batch],
        "article": [x["article"] for x in batch],
        "reference": [x["reference"] for x in batch],
    }


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.999)


class GaussianDiffusion:
    def __init__(self, timesteps: int, device: str):
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

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = self.sqrt_alphas_bar[t].view(-1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_omab * noise
        return x_t, noise

    def predict_x0_from_eps(self, x_t, t, eps_pred):
        sqrt_ab = self.sqrt_alphas_bar[t].view(-1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1)
        x0_pred = (x_t - sqrt_omab * eps_pred) / torch.clamp(sqrt_ab, min=1e-8)
        return x0_pred

    @torch.no_grad()
    def p_sample(self, denoiser, x_t, t: int, article_embeds, article_mask):
        bsz = x_t.size(0)
        t_batch = torch.full((bsz,), t, device=x_t.device, dtype=torch.long)

        eps_pred = denoiser(
            x_t=x_t,
            t=t_batch,
            article_embeds=article_embeds,
            article_mask=article_mask,
        )

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_bar[t]

        coef = beta_t / torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-8))
        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - coef * eps_pred)

        if t == 0:
            return mean

        noise = torch.randn_like(x_t)
        sigma = torch.sqrt(beta_t)
        return mean + sigma * noise

    @torch.no_grad()
    def sample(self, denoiser, shape, article_embeds, article_mask, sample_steps: int):
        x = torch.randn(shape, device=self.device)

        if sample_steps >= self.timesteps:
            timesteps = list(reversed(range(self.timesteps)))
        else:
            timesteps = torch.linspace(
                self.timesteps - 1,
                0,
                sample_steps,
                device=self.device,
            ).long().tolist()

        for t in timesteps:
            x = self.p_sample(
                denoiser=denoiser,
                x_t=x,
                t=int(t),
                article_embeds=article_embeds,
                article_mask=article_mask,
            )

        return x


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(0, half, device=device).float()
            / max(half, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class DiffusionLatentPlanner(nn.Module):
    def __init__(self, hidden_dim: int, n_layers: int = 4, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_emb = TimeEmbedding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.plan_type = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.article_type = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_t, t, article_embeds, article_mask):
        bsz, plan_len, _ = x_t.shape

        t_emb = self.time_emb(t).unsqueeze(1)
        x_t = x_t + t_emb + self.plan_type
        article_embeds = article_embeds + self.article_type

        h = torch.cat([x_t, article_embeds], dim=1)

        plan_mask = torch.ones(
            bsz,
            plan_len,
            dtype=article_mask.dtype,
            device=article_mask.device,
        )
        full_mask = torch.cat([plan_mask, article_mask], dim=1)
        src_key_padding_mask = full_mask == 0

        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        plan_h = h[:, :plan_len, :]
        eps_pred = self.out(self.out_norm(plan_h))
        return eps_pred


class LatentProjector(nn.Module):
    def __init__(self, hidden_dim: int, scale: float = 0.1):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Identity init: projector starts close to identity.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.scale * self.net(x)


class PP2LatentPlanSystem:
    def __init__(self, args):
        self.args = args
        self.device = DEVICE
        ensure_dir(args.out_dir)

        print("Device:", self.device)

        try:
            self.tok = AutoTokenizer.from_pretrained(args.llm_adapter, trust_remote_code=True)
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(args.base_llm, trust_remote_code=True)

        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        if args.llm_dtype == "bf16":
            dtype = torch.bfloat16
        elif args.llm_dtype == "fp32":
            dtype = torch.float32
        elif args.llm_dtype == "fp16":
            dtype = torch.float16 if self.device == "cuda" else torch.float32

        base_llm = AutoModelForCausalLM.from_pretrained(
            args.base_llm,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)

        self.llm = PeftModel.from_pretrained(base_llm, args.llm_adapter).to(self.device)
        self.llm.eval()
        self.llm.config.use_cache = False

        for p in self.llm.parameters():
            p.requires_grad = False

        self.hidden_dim = self.llm.config.hidden_size
        self.embed = self.llm.get_input_embeddings()

        self.diffusion = GaussianDiffusion(timesteps=args.t, device=self.device)

        self.dlm = DiffusionLatentPlanner(
            hidden_dim=self.hidden_dim,
            n_layers=args.dlm_layers,
            n_heads=args.dlm_heads,
            dropout=args.dropout,
        ).to(self.device)

        self.projector = LatentProjector(
            hidden_dim=self.hidden_dim,
            scale=args.projector_scale,
        ).to(self.device)

    def make_prompt(self, article: str):
        return f"""### Article:
{article}

### Task:
Write a concise and factual news summary in 2-3 sentences. Do not use bullet points.

### Summary:
"""

    def tokenize_article(self, articles: List[str]):
        tok = self.tok(
            articles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.max_article_len,
        ).to(self.device)
        return tok.input_ids, tok.attention_mask

    def tokenize_reference(self, references: List[str]):
        tok = self.tok(
            references,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.args.max_plan_len,
        ).to(self.device)
        return tok.input_ids, tok.attention_mask

    @torch.no_grad()
    def get_article_embeds(self, articles: List[str]):
        input_ids, attention_mask = self.tokenize_article(articles)
        embeds = self.embed(input_ids).float()
        return embeds, attention_mask

    @torch.no_grad()
    def get_gold_plan_embeds(self, references: List[str]):
        input_ids, attention_mask = self.tokenize_reference(references)
        embeds = self.embed(input_ids).float()
        embeds = embeds * attention_mask.unsqueeze(-1).float()
        return embeds, attention_mask, input_ids

    def masked_mse_loss(self, pred, target, mask):
        mask = mask.unsqueeze(-1).float()
        mask = mask.expand_as(pred)
        denom = torch.clamp(mask.sum(), min=1.0)
        return ((pred - target) ** 2 * mask).sum() / denom

    def save_ckpt(self, path: str):
        ensure_dir(os.path.dirname(path))
        torch.save({
            "dlm": self.dlm.state_dict(),
            "projector": self.projector.state_dict(),
            "hidden_dim": self.hidden_dim,
            "T": self.args.t,
            "max_plan_len": self.args.max_plan_len,
            "args": vars(self.args),
        }, path)
        print("Saved:", path)

    def load_ckpt(self, path: str, load_dlm=True, load_projector=True):
        ckpt = torch.load(path, map_location=self.device)

        if load_dlm and "dlm" in ckpt:
            self.dlm.load_state_dict(ckpt["dlm"])

        if load_projector:
            if "projector" in ckpt:
                self.projector.load_state_dict(ckpt["projector"])
            elif "bridge" in ckpt:
                self.projector.load_state_dict(ckpt["bridge"])

        print("Loaded:", path)

    def train_dlm(self):
        dataset = SummaryDataset(self.args.train_file, max_samples=self.args.max_train_samples)
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        self.dlm.train()
        self.projector.eval()

        optimizer = torch.optim.AdamW(
            self.dlm.parameters(),
            lr=self.args.lr_dlm,
            eps=1e-8,
            weight_decay=0.01,
        )

        optimizer.zero_grad()
        global_step = 0

        for epoch in range(self.args.epochs):
            total_loss = 0.0
            good_steps = 0
            pbar = tqdm(loader, desc=f"train_dlm epoch {epoch + 1}")

            for batch in pbar:
                with torch.no_grad():
                    article_embeds, article_mask = self.get_article_embeds(batch["article"])
                    z_plan, z_mask, _ = self.get_gold_plan_embeds(batch["reference"])

                bsz = z_plan.size(0)
                t = torch.randint(0, self.args.t, (bsz,), device=self.device, dtype=torch.long)
                x_t, noise = self.diffusion.q_sample(z_plan, t)

                eps_pred = self.dlm(
                    x_t=x_t,
                    t=t,
                    article_embeds=article_embeds,
                    article_mask=article_mask,
                )

                loss = self.masked_mse_loss(pred=eps_pred, target=noise, mask=z_mask)

                if not torch.isfinite(loss):
                    print("Skip NaN/Inf DLM batch")
                    optimizer.zero_grad()
                    continue

                (loss / self.args.grad_accum).backward()

                if (global_step + 1) % self.args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.dlm.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item()
                good_steps += 1
                global_step += 1
                pbar.set_postfix({"loss_diff": f"{loss.item():.4f}"})

            avg_loss = total_loss / max(good_steps, 1)
            print(f"train_dlm epoch={epoch + 1} avg_loss={avg_loss:.4f}")
            self.save_ckpt(os.path.join(self.args.out_dir, f"dlm_epoch_{epoch + 1}.pt"))

        self.save_ckpt(os.path.join(self.args.out_dir, "dlm_final.pt"))

    def build_lm_inputs_with_soft_prefix(self, articles, references, soft_prefix_embeds):
        prompts = [self.make_prompt(a) for a in articles]
        answers = [r.strip() + self.tok.eos_token for r in references]

        max_answer_len = self.args.max_answer_len
        max_prompt_len = self.args.llm_max_len - max_answer_len

        all_input_ids = []
        all_attention_mask = []
        all_labels = []

        for prompt, answer in zip(prompts, answers):
            p_tok = self.tok(
                prompt,
                truncation=True,
                max_length=max_prompt_len,
                padding=False,
            )

            a_tok = self.tok(
                answer,
                truncation=True,
                max_length=max_answer_len,
                padding=False,
            )

            input_ids = p_tok["input_ids"] + a_tok["input_ids"]
            attention_mask = [1] * len(input_ids)
            labels = [-100] * len(p_tok["input_ids"]) + a_tok["input_ids"]

            pad_len = self.args.llm_max_len - len(input_ids)
            if pad_len > 0:
                input_ids += [self.tok.pad_token_id] * pad_len
                attention_mask += [0] * pad_len
                labels += [-100] * pad_len

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)

        input_ids = torch.tensor(all_input_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(all_attention_mask, dtype=torch.long, device=self.device)
        labels = torch.tensor(all_labels, dtype=torch.long, device=self.device)

        token_embeds = self.embed(input_ids)
        soft_prefix_embeds = soft_prefix_embeds.to(token_embeds.dtype)

        inputs_embeds = torch.cat([soft_prefix_embeds, token_embeds], dim=1)

        prefix_mask = torch.ones(
            soft_prefix_embeds.shape[:2],
            dtype=attention_mask.dtype,
            device=self.device,
        )
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        prefix_labels = torch.full(
            soft_prefix_embeds.shape[:2],
            -100,
            dtype=labels.dtype,
            device=self.device,
        )
        labels = torch.cat([prefix_labels, labels], dim=1)

        return inputs_embeds, attention_mask, labels

    @torch.no_grad()
    def get_dlm_predicted_z0_for_projector(self, articles, references):
        article_embeds, article_mask = self.get_article_embeds(articles)
        z_plan, z_mask, _ = self.get_gold_plan_embeds(references)

        bsz = z_plan.size(0)
        t_max = min(self.args.t, self.args.projector_t_max)
        t = torch.randint(0, t_max, (bsz,), device=self.device, dtype=torch.long)

        x_t, _ = self.diffusion.q_sample(z_plan, t)

        eps_pred = self.dlm(
            x_t=x_t,
            t=t,
            article_embeds=article_embeds,
            article_mask=article_mask,
        )

        z0_pred = self.diffusion.predict_x0_from_eps(x_t, t, eps_pred)

        z0_pred = torch.nan_to_num(
            z0_pred,
            nan=0.0,
            posinf=self.args.latent_clip,
            neginf=-self.args.latent_clip,
        )
        z0_pred = torch.clamp(z0_pred, -self.args.latent_clip, self.args.latent_clip)

        z0_pred = z0_pred * z_mask.unsqueeze(-1).float()
        return z0_pred, z_mask

    def train_projector(self):
        if not self.args.ckpt:
            raise ValueError("--ckpt pointing to dlm_final.pt is required for train_projector")

        self.load_ckpt(self.args.ckpt, load_dlm=True, load_projector=False)

        dataset = SummaryDataset(self.args.train_file, max_samples=self.args.max_train_samples)
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        self.dlm.eval()
        for p in self.dlm.parameters():
            p.requires_grad = False

        self.projector.train()

        optimizer = torch.optim.AdamW(
            self.projector.parameters(),
            lr=self.args.lr_projector,
            eps=1e-8,
            weight_decay=0.01,
        )

        optimizer.zero_grad()
        global_step = 0

        for epoch in range(self.args.epochs):
            total_loss = 0.0
            good_steps = 0
            pbar = tqdm(loader, desc=f"train_projector epoch {epoch + 1}")

            for batch in pbar:
                with torch.no_grad():
                    if self.args.projector_input == "gold":
                        z_prefix, _, _ = self.get_gold_plan_embeds(batch["reference"])
                    elif self.args.projector_input == "dlm_pred":
                        z_prefix, _ = self.get_dlm_predicted_z0_for_projector(
                            batch["article"],
                            batch["reference"],
                        )
                    else:
                        raise ValueError(f"Unknown projector_input: {self.args.projector_input}")

                z_prefix = torch.nan_to_num(
                    z_prefix,
                    nan=0.0,
                    posinf=self.args.latent_clip,
                    neginf=-self.args.latent_clip,
                )
                z_prefix = torch.clamp(z_prefix, -self.args.latent_clip, self.args.latent_clip)

                if not torch.isfinite(z_prefix).all():
                    print("Skip non-finite z_prefix")
                    optimizer.zero_grad()
                    continue

                soft_prefix = self.projector(z_prefix)

                soft_prefix = torch.nan_to_num(
                    soft_prefix,
                    nan=0.0,
                    posinf=self.args.latent_clip,
                    neginf=-self.args.latent_clip,
                )
                soft_prefix = torch.clamp(
                    soft_prefix,
                    -self.args.latent_clip,
                    self.args.latent_clip,
                )

                inputs_embeds, attention_mask, labels = self.build_lm_inputs_with_soft_prefix(
                    batch["article"],
                    batch["reference"],
                    soft_prefix,
                )

                if not torch.isfinite(inputs_embeds).all():
                    print("Skip non-finite inputs_embeds")
                    optimizer.zero_grad()
                    continue

                if (labels != -100).sum().item() == 0:
                    print("Skip empty labels")
                    optimizer.zero_grad()
                    continue

                outputs = self.llm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )

                loss = outputs.loss

                if not torch.isfinite(loss):
                    print(
                        "Skip NaN/Inf projector batch",
                        "soft_min", soft_prefix.min().item(),
                        "soft_max", soft_prefix.max().item(),
                        "soft_std", soft_prefix.std().item(),
                    )
                    optimizer.zero_grad()
                    continue

                (loss / self.args.grad_accum).backward()

                if (global_step + 1) % self.args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.projector.parameters(), 0.5)
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item()
                good_steps += 1
                global_step += 1
                pbar.set_postfix({"loss_lm": f"{loss.item():.4f}"})

            avg_loss = total_loss / max(good_steps, 1)
            print(f"train_projector epoch={epoch + 1} avg_loss={avg_loss:.4f}")
            self.save_ckpt(os.path.join(self.args.out_dir, f"projector_epoch_{epoch + 1}.pt"))

        self.save_ckpt(os.path.join(self.args.out_dir, "final.pt"))

    @torch.no_grad()
    def sample_latent_plan(self, article: str):
        self.dlm.eval()
        article_embeds, article_mask = self.get_article_embeds([article])

        shape = (1, self.args.max_plan_len, self.hidden_dim)

        z_plan = self.diffusion.sample(
            denoiser=self.dlm,
            shape=shape,
            article_embeds=article_embeds,
            article_mask=article_mask,
            sample_steps=self.args.sample_steps,
        )

        z_plan = torch.nan_to_num(
            z_plan,
            nan=0.0,
            posinf=self.args.latent_clip,
            neginf=-self.args.latent_clip,
        )
        z_plan = torch.clamp(z_plan, -self.args.latent_clip, self.args.latent_clip)

        return z_plan

    @torch.no_grad()
    def generate_from_latent_plan(self, article: str, z_plan):
        self.projector.eval()
        self.llm.eval()

        z_plan = torch.nan_to_num(
            z_plan,
            nan=0.0,
            posinf=self.args.latent_clip,
            neginf=-self.args.latent_clip,
        )
        z_plan = torch.clamp(z_plan, -self.args.latent_clip, self.args.latent_clip)

        soft_prefix = self.projector(z_plan)

        soft_prefix = torch.nan_to_num(
            soft_prefix,
            nan=0.0,
            posinf=self.args.latent_clip,
            neginf=-self.args.latent_clip,
        )
        soft_prefix = torch.clamp(soft_prefix, -self.args.latent_clip, self.args.latent_clip)

        prompt = self.make_prompt(article)

        tok = self.tok(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.llm_max_len,
        ).to(self.device)

        prompt_embeds = self.embed(tok.input_ids)
        soft_prefix = soft_prefix.to(prompt_embeds.dtype)

        inputs_embeds = torch.cat([soft_prefix, prompt_embeds], dim=1)

        prefix_mask = torch.ones(
            soft_prefix.shape[:2],
            dtype=tok.attention_mask.dtype,
            device=self.device,
        )
        attention_mask = torch.cat([prefix_mask, tok.attention_mask], dim=1)

        generated = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.args.max_new_tokens,
            min_new_tokens=self.args.min_new_tokens,
            num_beams=self.args.num_beams,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=self.args.repetition_penalty,
            length_penalty=self.args.length_penalty,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.pad_token_id,
        )

        text = self.tok.decode(generated[0], skip_special_tokens=True)

        return clean_summary(
            text,
            max_sentences=self.args.max_sentences,
            max_words=self.args.max_words,
        )

    @torch.no_grad()
    def generate_one(self, article: str):
        z_plan = self.sample_latent_plan(article)
        pred = self.generate_from_latent_plan(article, z_plan)
        return pred

    def infer(self):
        if not self.args.ckpt:
            raise ValueError("--ckpt is required for infer")

        self.load_ckpt(self.args.ckpt, load_dlm=True, load_projector=True)

        dataset = SummaryDataset(self.args.test_file, max_samples=self.args.max_test_samples)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        ensure_dir(os.path.dirname(self.args.output_jsonl) or ".")

        with open(self.args.output_jsonl, "w", encoding="utf-8") as fout:
            for batch in tqdm(loader, desc="infer"):
                article = batch["article"][0]
                pred = self.generate_one(article)

                row = {
                    "id": batch["id"][0],
                    "article": article,
                    "reference": batch["reference"][0],
                    "pp2_summary": pred,
                    "method": "dlm_latent_plan_projector_llm_decoder",
                }

                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

        print("Saved:", self.args.output_jsonl)

    def oracle_infer(self):
        if self.args.ckpt:
            self.load_ckpt(self.args.ckpt, load_dlm=False, load_projector=True)

        dataset = SummaryDataset(self.args.test_file, max_samples=self.args.max_test_samples)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        ensure_dir(os.path.dirname(self.args.output_jsonl) or ".")

        with open(self.args.output_jsonl, "w", encoding="utf-8") as fout:
            for batch in tqdm(loader, desc="oracle_infer"):
                article = batch["article"][0]

                z_gold, _, _ = self.get_gold_plan_embeds(batch["reference"])
                pred = self.generate_from_latent_plan(article, z_gold)

                row = {
                    "id": batch["id"][0],
                    "article": article,
                    "reference": batch["reference"][0],
                    "pp2_summary": pred,
                    "method": "oracle_gold_latent_plan_projector_llm_decoder",
                }

                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

        print("Saved:", self.args.output_jsonl)


def build_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--mode",
        choices=["train_dlm", "train_projector", "infer", "oracle_infer"],
        required=True,
    )

    p.add_argument("--base_llm", default=BASE_LLM)
    p.add_argument("--llm_adapter", default=LLM_ADAPTER)

    p.add_argument("--train_file", default=TRAIN_FILE)
    p.add_argument("--test_file", default=TEST_FILE)
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--ckpt", default=None)

    p.add_argument(
        "--output_jsonl",
        default=os.path.join(OUT_DIR, "test_pp2_latent_plan_projector.jsonl"),
    )

    p.add_argument("--max_article_len", type=int, default=MAX_ARTICLE_LEN)
    p.add_argument("--max_plan_len", type=int, default=MAX_PLAN_LEN)
    p.add_argument("--llm_max_len", type=int, default=LLM_MAX_LEN)

    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    p.add_argument("--epochs", type=int, default=EPOCHS)

    p.add_argument("--lr_dlm", type=float, default=LR_DLM)
    p.add_argument("--lr_projector", type=float, default=LR_PROJECTOR)

    p.add_argument("--t", type=int, default=T)
    p.add_argument("--sample_steps", type=int, default=SAMPLE_STEPS)

    p.add_argument("--latent_clip", type=float, default=2.0)
    p.add_argument("--projector_t_max", type=int, default=500)
    p.add_argument("--projector_scale", type=float, default=0.1)

    p.add_argument("--dlm_layers", type=int, default=4)
    p.add_argument("--dlm_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--max_train_samples", type=int, default=20000)
    p.add_argument("--max_test_samples", type=int, default=None)

    p.add_argument("--projector_input", choices=["dlm_pred", "gold"], default="dlm_pred")
    p.add_argument("--llm_dtype", choices=["fp16", "bf16", "fp32"], default="fp16")

    p.add_argument("--max_new_tokens", type=int, default=90)
    p.add_argument("--min_new_tokens", type=int, default=25)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--repetition_penalty", type=float, default=1.15)
    p.add_argument("--length_penalty", type=float, default=0.8)
    p.add_argument("--max_sentences", type=int, default=3)
    p.add_argument("--max_words", type=int, default=90)

    p.add_argument("--seed", type=int, default=102)
    p.add_argument("--max_answer_len", type=int, default=160)

    return p.parse_args()


def main():
    args = build_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    system = PP2LatentPlanSystem(args)

    if args.mode == "train_dlm":
        system.train_dlm()
    elif args.mode == "train_projector":
        system.train_projector()
    elif args.mode == "infer":
        system.infer()
    elif args.mode == "oracle_infer":
        system.oracle_infer()


if __name__ == "__main__":
    main()