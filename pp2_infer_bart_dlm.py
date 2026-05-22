import json, torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BartTokenizer, BartModel
from peft import PeftModel

# =====================
# CONFIG
# =====================
BASE_LLM = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_ADAPTER = "outputs/qwen_summarizer/final"

BART_NAME = "facebook/bart-base"
CKPT = "outputs/pp2_bart_dlm/final.pt"

INPUT_FILE = "data/test.jsonl"
OUTPUT_FILE = "data/test_pp2.jsonl"

MAX_ARTICLE_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SemanticDLM(nn.Module):
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
# LOAD BART
# =====================
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
        max_length=max_length,
    ).to(DEVICE)

    outputs = bart.encoder(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
    )

    h = outputs.last_hidden_state
    mask = inputs.attention_mask.unsqueeze(-1)
    z = (h * mask).sum(dim=1) / mask.sum(dim=1)
    return z.float()


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

ckpt = torch.load(CKPT, map_location=DEVICE)

LLM_DIM = ckpt["llm_dim"]
PREFIX_LEN = ckpt["prefix_len"]

dlm = SemanticDLM(dim=768).to(DEVICE)
projector = PrefixProjector(
    in_dim=768,
    llm_dim=LLM_DIM,
    prefix_len=PREFIX_LEN
).to(DEVICE)

dlm.load_state_dict(ckpt["dlm"])
projector.load_state_dict(ckpt["projector"])

dlm.eval()
projector.eval()


def make_prompt(article):
    return f"""### Article:
{article}

### Task:
Summarize the article concisely and factually.

### Summary:
"""


@torch.no_grad()
def generate_pp2(article):
    # 1. article semantic embedding
    z_article = encode_bart([article], MAX_ARTICLE_LEN)

    # 2. DLM semantic reorganization
    z_summary = dlm(z_article)

    # 3. latent prefix
    latent_prefix = projector(z_summary)

    prompt = make_prompt(article)

    inputs = llm_tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=900,
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
        pad_token_id=llm_tok.eos_token_id,
    )

    text = llm_tok.decode(outputs[0], skip_special_tokens=True)

    if "### Summary:" in text:
        text = text.split("### Summary:")[-1].strip()

    return text


with open(INPUT_FILE, encoding="utf-8") as f, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    for line in tqdm(f):
        ex = json.loads(line)
        ex["pp2_summary"] = generate_pp2(ex["article"])
        out.write(json.dumps(ex, ensure_ascii=False) + "\n")

print("Saved:", OUTPUT_FILE)