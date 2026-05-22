import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
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

CKPT = "outputs/pp2_true_diffusion/final.pt"

INPUT_FILE = "data/test.jsonl"
OUTPUT_FILE = "data/test_pp2_true_diffusion.jsonl"

MAX_ARTICLE_LEN = 512
SAMPLE_STEPS = 1000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# DIFFUSION
# =========================================================

def cosine_beta_schedule(timesteps, s=0.008):
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

    @torch.no_grad()
    def p_sample(self, denoiser, z_t, t, z_article):
        batch_size = z_t.size(0)

        t_batch = torch.full(
            (batch_size,),
            t,
            device=z_t.device,
            dtype=torch.long
        )

        eps_pred = denoiser(
            z_t=z_t,
            t=t_batch,
            z_article=z_article
        )

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_bar[t]

        coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            z_t - coef * eps_pred
        )

        if t == 0:
            return mean

        noise = torch.randn_like(z_t)
        sigma = torch.sqrt(beta_t)

        return mean + sigma * noise

    @torch.no_grad()
    def sample(self, denoiser, z_article, shape):
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
        half = self.dim // 2
        device = t.device

        freqs = torch.exp(
            -math.log(10000) *
            torch.arange(0, half, device=device).float() / half
        )

        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)

        emb = torch.cat(
            [torch.sin(args), torch.cos(args)],
            dim=-1
        )

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return self.mlp(emb)


class ConditionalDenoiser(nn.Module):
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

            nn.Linear(dim * 4, dim)
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
    def __init__(self, in_dim=768, llm_dim=1536, prefix_len=8):
        super().__init__()

        self.prefix_len = prefix_len
        self.llm_dim = llm_dim

        self.proj = nn.Sequential(
            nn.Linear(in_dim, llm_dim),
            nn.Tanh(),
            nn.Linear(llm_dim, prefix_len * llm_dim)
        )

    def forward(self, z):
        prefix = self.proj(z)

        return prefix.view(
            z.size(0),
            self.prefix_len,
            self.llm_dim
        )


# =========================================================
# LOAD BART
# =========================================================

bart_tok = BartTokenizer.from_pretrained(BART_NAME)
bart = BartModel.from_pretrained(BART_NAME).to(DEVICE)
bart.eval()


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
# LOAD LLM
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


# =========================================================
# LOAD CKPT
# =========================================================

ckpt = torch.load(CKPT, map_location=DEVICE)

LLM_DIM = ckpt["llm_dim"]
PREFIX_LEN = ckpt["prefix_len"]
T = ckpt["T"]

denoiser = ConditionalDenoiser(dim=768).to(DEVICE)
projector = PrefixProjector(
    in_dim=768,
    llm_dim=LLM_DIM,
    prefix_len=PREFIX_LEN
).to(DEVICE)

denoiser.load_state_dict(ckpt["denoiser"])
projector.load_state_dict(ckpt["projector"])

denoiser.eval()
projector.eval()

diffusion = GaussianDiffusion(
    timesteps=T,
    device=DEVICE
)


# =========================================================
# GENERATE
# =========================================================

def make_prompt(article):
    return f"""### Article:
{article}

### Task:
Summarize the article concisely and factually.

### Summary:
"""


@torch.no_grad()
def generate_pp2(article):
    z_article = encode_bart(
        [article],
        MAX_ARTICLE_LEN
    )

    z_summary = diffusion.sample(
        denoiser=denoiser,
        z_article=z_article,
        shape=z_article.shape
    )

    latent_prefix = projector(z_summary)

    prompt = make_prompt(article)

    inputs = llm_tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=900
    ).to(DEVICE)

    token_embeds = llm.get_input_embeddings()(inputs.input_ids)

    latent_prefix = latent_prefix.to(token_embeds.dtype)

    inputs_embeds = torch.cat(
        [latent_prefix, token_embeds],
        dim=1
    )

    prefix_mask = torch.ones(
        latent_prefix.shape[:2],
        dtype=inputs.attention_mask.dtype,
        device=DEVICE
    )

    attention_mask = torch.cat(
        [prefix_mask, inputs.attention_mask],
        dim=1
    )

    outputs = llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=128,
        num_beams=4,
        do_sample=False,
        pad_token_id=llm_tok.eos_token_id
    )

    text = llm_tok.decode(
        outputs[0],
        skip_special_tokens=True
    )

    if "### Summary:" in text:
        text = text.split("### Summary:")[-1].strip()

    return text


with open(INPUT_FILE, encoding="utf-8") as f, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    for line in tqdm(f):
        ex = json.loads(line)

        ex["pp2_summary"] = generate_pp2(ex["article"])

        out.write(
            json.dumps(ex, ensure_ascii=False) + "\n"
        )

print("Saved:", OUTPUT_FILE)

# article → BART z_article
# noise z_T → reverse diffusion conditioned on z_article → z_summary
# z_summary → projector → latent prefix
# Qwen/LLaMA → summary