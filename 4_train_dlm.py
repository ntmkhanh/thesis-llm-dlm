import json, math, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

<<<<<<< HEAD
# MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
=======
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
>>>>>>> deb03ea (8b)
DEVICE = "cuda"

TIMESTEPS = 200
BATCH_SIZE = 4
EPOCHS = 5
LR = 1e-4


class DraftDataset(Dataset):
    def __init__(self, path, max_samples=None):
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
        "draft": [x["draft"] for x in batch],
        "reference": [x["reference"] for x in batch],
    }


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return self.mlp(emb)


class ResidualDDPM(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.time_emb = TimeEmbedding(hidden_size)

        self.net = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, r_t, t, z_draft, z_article):
        t_emb = self.time_emb(t)
        x = torch.cat([r_t, z_draft, z_article, t_emb], dim=-1)
        pred_noise = self.net(x)
        return pred_noise


class DiffusionSchedule:
    def __init__(self, timesteps, device):
        self.timesteps = timesteps

        betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def q_sample(self, r_0, t, noise):
        sqrt_ab = self.sqrt_alpha_bars[t].unsqueeze(-1)
        sqrt_omab = self.sqrt_one_minus_alpha_bars[t].unsqueeze(-1)
        return sqrt_ab * r_0 + sqrt_omab * noise


@torch.no_grad()
def encode_text(model, tokenizer, texts, max_length=256):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(DEVICE)

    outputs = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
    )

    h = outputs.hidden_states[-1]
    mask = inputs.attention_mask.unsqueeze(-1)
    pooled = (h * mask).sum(dim=1) / mask.sum(dim=1)

    return pooled.float()


@torch.no_grad()
def sample_residual(ddpm, schedule, z_draft, z_article):
    ddpm.eval()

    r_t = torch.randn_like(z_draft)

    for step in reversed(range(schedule.timesteps)):
        t = torch.full(
            (z_draft.size(0),),
            step,
            device=z_draft.device,
            dtype=torch.long,
        )

        pred_noise = ddpm(r_t, t, z_draft, z_article)

        beta_t = schedule.betas[t].unsqueeze(-1)
        alpha_t = schedule.alphas[t].unsqueeze(-1)
        alpha_bar_t = schedule.alpha_bars[t].unsqueeze(-1)

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            r_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
        )

        if step > 0:
            noise = torch.randn_like(r_t)
            r_t = mean + torch.sqrt(beta_t) * noise
        else:
            r_t = mean

    r_0 = r_t
    return r_0


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

llama = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)
llama.eval()

hidden = llama.config.hidden_size

ddpm = ResidualDDPM(hidden).to(DEVICE)
schedule = DiffusionSchedule(TIMESTEPS, DEVICE)

optimizer = torch.optim.AdamW(ddpm.parameters(), lr=LR)

<<<<<<< HEAD
train_ds = DraftDataset("data/train_drafts_5000.jsonl", max_samples=20000)
loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate)

for epoch in range(100):
    dlm.train()
    total = 1
=======
train_ds = DraftDataset("data/train_drafts.jsonl", max_samples=20000)
loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate,
)

for epoch in range(EPOCHS):
    ddpm.train()
    total = 0.0
>>>>>>> deb03ea (8b)

    for batch in tqdm(loader):
        with torch.no_grad():
            z_draft = encode_text(llama, tokenizer, batch["draft"], 256)
            z_ref = encode_text(llama, tokenizer, batch["reference"], 256)
            z_article = encode_text(llama, tokenizer, batch["article"], 512)

        # Residual latent
        r_0 = z_ref - z_draft

        # Random timestep
        t = torch.randint(
            0,
            TIMESTEPS,
            (z_draft.size(0),),
            device=DEVICE,
            dtype=torch.long,
        )

        # Forward diffusion on residual
        noise = torch.randn_like(r_0)
        r_t = schedule.q_sample(r_0, t, noise)

        # Noise prediction
        pred_noise = ddpm(r_t, t, z_draft, z_article)

        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()

    print(f"epoch={epoch} loss={total / len(loader):.4f}")

<<<<<<< HEAD
torch.save(dlm.state_dict(), "outputs/residual_dlm_100.pt")

# train Residual DLM 
# z_refined = z_draft + Δz
=======
torch.save(ddpm.state_dict(), "outputs/residual_ddpm.pt")
>>>>>>> deb03ea (8b)
