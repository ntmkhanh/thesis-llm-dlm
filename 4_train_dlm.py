import json, torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda"

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
        "reference": [x["reference"] for x in batch]
    }

class ResidualDLM(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, z_draft_noisy, z_article):
        x = torch.cat([z_draft_noisy, z_article], dim=-1)
        delta = self.net(x)
        return delta

def add_noise(z, sigma=0.1):
    return z + torch.randn_like(z) * sigma

@torch.no_grad()
def encode_text(model, tokenizer, texts, max_length=256):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(DEVICE)

    outputs = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False
    )

    h = outputs.hidden_states[-1]          # [B, T, H]
    mask = inputs.attention_mask.unsqueeze(-1)
    pooled = (h * mask).sum(dim=1) / mask.sum(dim=1)
    return pooled.float()                  # [B, H]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

llama = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
llama.eval()

hidden = llama.config.hidden_size
dlm = ResidualDLM(hidden).to(DEVICE)

optimizer = torch.optim.AdamW(dlm.parameters(), lr=1e-4)

train_ds = DraftDataset("data/train_drafts.jsonl", max_samples=20000)
loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate)

for epoch in range(5):
    dlm.train()
    total = 0

    for batch in tqdm(loader):
        with torch.no_grad():
            z_draft = encode_text(llama, tokenizer, batch["draft"], 256)
            z_ref = encode_text(llama, tokenizer, batch["reference"], 256)
            z_article = encode_text(llama, tokenizer, batch["article"], 512)

        z_noisy = add_noise(z_draft, sigma=0.1)

        delta = dlm(z_noisy, z_article)
        z_refined = z_draft + delta

        loss_mse = F.mse_loss(z_refined, z_ref)
        loss_cos = 1 - F.cosine_similarity(z_refined, z_ref, dim=-1).mean()

        loss = loss_mse + 0.1 * loss_cos

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()

    print(f"epoch={epoch} loss={total/len(loader):.4f}")

torch.save(dlm.state_dict(), "outputs/residual_dlm.pt")

# train Residual DLM 
# z_refined = z_draft + Δz