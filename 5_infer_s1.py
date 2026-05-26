import json, torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "outputs/qwen_summarizer/final"
DEVICE = "cuda"

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

class LatentProjector(nn.Module):
    def __init__(self, hidden_size, prefix_len=8):
        super().__init__()
        self.prefix_len = prefix_len
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, prefix_len * hidden_size)
        )

    def forward(self, z):
        p = self.proj(z)
        return p.view(z.size(0), self.prefix_len, z.size(-1))

@torch.no_grad()
def encode_text(model, tokenizer, texts, max_length):
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

    h = outputs.hidden_states[-1]
    mask = inputs.attention_mask.unsqueeze(-1)
    pooled = (h * mask).sum(dim=1) / mask.sum(dim=1)
    return pooled.float()

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

llama = PeftModel.from_pretrained(base, ADAPTER)
llama.eval()

hidden = llama.config.hidden_size

dlm = ResidualDLM(hidden).to(DEVICE)
dlm.load_state_dict(torch.load("outputs/residual_dlm_100.pt"))
dlm.eval()

projector = LatentProjector(hidden, prefix_len=8).to(DEVICE)

def final_prompt(article, draft):
    return f"""### Article:
{article}

### Draft Summary:
{draft}

### Task:
Improve the draft summary. Keep it factual, concise, and faithful to the article.

### Final Summary:
"""

@torch.no_grad()
def generate_s1(article, draft):
    z_draft = encode_text(llama, tokenizer, [draft], 256)
    z_article = encode_text(llama, tokenizer, [article], 512)

    delta = dlm(z_draft, z_article)
    z_refined = z_draft + delta

    latent_prefix = projector(z_refined).to(torch.float16)

    inputs = tokenizer(
        final_prompt(article, draft),
        return_tensors="pt",
        truncation=True,
        max_length=900
    ).to(DEVICE)

    token_embeds = llama.get_input_embeddings()(inputs.input_ids)

    inputs_embeds = torch.cat([latent_prefix, token_embeds], dim=1)

    prefix_mask = torch.ones(
        latent_prefix.shape[:2],
        dtype=inputs.attention_mask.dtype,
        device=DEVICE
    )

    attention_mask = torch.cat([prefix_mask, inputs.attention_mask], dim=1)

    outputs = llama.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=128,
        num_beams=4,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("### Final Summary:")[-1].strip()

with open("data/test_drafts_500.jsonl", encoding="utf-8") as f, \
     open("data/test_s1_100.jsonl", "w", encoding="utf-8") as out:

    for line in tqdm(f):
        ex = json.loads(line)
        ex["s1"] = generate_s1(ex["article"], ex["draft"])
        out.write(json.dumps(ex, ensure_ascii=False) + "\n")


# S1 = LLaMA + DLM
# dùng z_refined làm latent prefix để LLaMA sinh summảy cuối cùng.
