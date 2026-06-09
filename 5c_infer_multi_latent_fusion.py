import json
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default="outputs/qwen_summarizer/final")

    p.add_argument("--input", default="data/test_multi_drafts_500.jsonl")
    p.add_argument("--output", default="data/test_multi_fusion_500.jsonl")

    p.add_argument("--dlm_path", default="outputs/residual_dlm.pt")

    p.add_argument(
        "--fusion_key",
        default="mean",
        choices=[
            "mean",
            "max",
            "min",
            "first",
            "last",
            "summac_weighted",
            "attention"
        ]
    )

    p.add_argument("--output_key", default=None)

    p.add_argument("--prefix_len", type=int, default=8)
    p.add_argument("--max_input_tokens", type=int, default=900)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--num_beams", type=int, default=4)

    return p.parse_args()


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

    def forward(self, z_draft, z_article):
        x = torch.cat([z_draft, z_article], dim=-1)
        return self.net(x)


class LatentProjector(nn.Module):
    def __init__(self, in_dim, llm_dim, prefix_len=8):
        super().__init__()

        self.prefix_len = prefix_len
        self.llm_dim = llm_dim

        self.proj = nn.Sequential(
            nn.Linear(in_dim, llm_dim),
            nn.Tanh(),
            nn.Linear(llm_dim, prefix_len * llm_dim)
        )

    def forward(self, z):
        p = self.proj(z)
        return p.view(z.size(0), self.prefix_len, self.llm_dim)


class AttentionFusion(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, z_drafts):
        scores = self.score(z_drafts)          # [K, 1]
        weights = torch.softmax(scores, dim=0) # [K, 1]
        z_fused = (weights * z_drafts).sum(dim=0, keepdim=True)

        return z_fused, weights.squeeze(-1)


@torch.no_grad()
def encode_llm(llm, tokenizer, texts, max_length, device):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)

    outputs = llm(
        **inputs,
        output_hidden_states=True,
        use_cache=False
    )

    h = outputs.hidden_states[-1]  # [B, T, H]
    mask = inputs.attention_mask.unsqueeze(-1)

    z = (h * mask).sum(dim=1) / mask.sum(dim=1)

    return z.float()


def fuse_latents(z_drafts, ex, fusion_key, attention_fusion=None):
    info = {}

    if fusion_key == "mean":
        z_fused = z_drafts.mean(dim=0, keepdim=True)
        info["fusion_weights"] = None
        return z_fused, info

    if fusion_key == "max":
        z_fused = z_drafts.max(dim=0, keepdim=True).values
        info["fusion_weights"] = None
        return z_fused, info

    if fusion_key == "min":
        z_fused = z_drafts.min(dim=0, keepdim=True).values
        info["fusion_weights"] = None
        return z_fused, info

    if fusion_key == "first":
        z_fused = z_drafts[0:1]
        info["fusion_weights"] = None
        return z_fused, info

    if fusion_key == "last":
        z_fused = z_drafts[-1:]
        info["fusion_weights"] = None
        return z_fused, info

    if fusion_key == "summac_weighted":
        scores = ex.get("draft_scores", None)

        if scores is None:
            raise ValueError(
                "fusion_key=summac_weighted cần input có key draft_scores. "
                "Hãy chạy 5b_select_best_draft.py trước."
            )

        scores = torch.tensor(
            scores,
            dtype=z_drafts.dtype,
            device=z_drafts.device
        )

        if scores.numel() != z_drafts.size(0):
            raise ValueError(
                f"Số draft_scores ({scores.numel()}) khác số drafts ({z_drafts.size(0)})"
            )

        weights = torch.softmax(scores, dim=0).view(-1, 1)
        z_fused = (weights * z_drafts).sum(dim=0, keepdim=True)

        info["fusion_weights"] = weights.squeeze(-1).detach().cpu().tolist()
        return z_fused, info

    if fusion_key == "attention":
        if attention_fusion is None:
            raise ValueError("fusion_key=attention nhưng attention_fusion=None")

        z_fused, weights = attention_fusion(z_drafts)
        info["fusion_weights"] = weights.detach().cpu().tolist()
        return z_fused, info

    raise ValueError(f"Unknown fusion_key: {fusion_key}")


def make_prompt(article, drafts):
    joined = "\n".join(
        [f"Draft {i + 1}: {d}" for i, d in enumerate(drafts)]
    )

    return f"""### Article:
{article}

### Candidate Drafts:
{joined}

### Task:
Write one concise, factual final summary based on the article and candidate drafts.

### Final Summary:
"""


def clean(text):
    if "### Final Summary:" in text:
        text = text.split("### Final Summary:")[-1]
    return text.strip()


@torch.no_grad()
def generate_one(
    ex,
    args,
    llm,
    tokenizer,
    dlm,
    projector,
    attention_fusion,
    device
):
    article = ex["article"]
    drafts = ex["drafts"]

    z_drafts = encode_llm(
        llm=llm,
        tokenizer=tokenizer,
        texts=drafts,
        max_length=256,
        device=device
    )

    z_fused, fusion_info = fuse_latents(
        z_drafts=z_drafts,
        ex=ex,
        fusion_key=args.fusion_key,
        attention_fusion=attention_fusion
    )

    z_article = encode_llm(
        llm=llm,
        tokenizer=tokenizer,
        texts=[article],
        max_length=512,
        device=device
    )

    delta = dlm(z_fused, z_article)
    z_refined = z_fused + delta

    latent_prefix = projector(z_refined)

    inputs = tokenizer(
        make_prompt(article, drafts),
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_tokens
    ).to(device)

    token_embeds = llm.get_input_embeddings()(inputs.input_ids)
    latent_prefix = latent_prefix.to(token_embeds.dtype)

    inputs_embeds = torch.cat(
        [latent_prefix, token_embeds],
        dim=1
    )

    prefix_mask = torch.ones(
        latent_prefix.shape[:2],
        dtype=inputs.attention_mask.dtype,
        device=device
    )

    attention_mask = torch.cat(
        [prefix_mask, inputs.attention_mask],
        dim=1
    )

    outputs = llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return clean(text), fusion_info


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_key = args.output_key
    if output_key is None:
        output_key = f"multi_fusion_{args.fusion_key}"

    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    llm = PeftModel.from_pretrained(base, args.adapter)
    llm.eval()

    llm_dim = llm.config.hidden_size
    print("LLM hidden size:", llm_dim)

    dlm = ResidualDLM(hidden_size=llm_dim).to(device)

    try:
        state = torch.load(
            args.dlm_path,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        state = torch.load(
            args.dlm_path,
            map_location=device
        )

    dlm.load_state_dict(state)
    dlm.eval()

    projector = LatentProjector(
        in_dim=llm_dim,
        llm_dim=llm_dim,
        prefix_len=args.prefix_len
    ).to(device)
    projector.eval()

    attention_fusion = None

    if args.fusion_key == "attention":
        attention_fusion = AttentionFusion(
            hidden_size=llm_dim
        ).to(device)

        attention_fusion.eval()

        print(
            "WARNING: AttentionFusion chưa được train. "
            "Kết quả attention chỉ dùng để test kỹ thuật."
        )

    with open(args.input, encoding="utf-8") as f, \
         open(args.output, "w", encoding="utf-8") as out:

        for line in tqdm(f, desc=f"Fusion={args.fusion_key}"):
            ex = json.loads(line)

            summary, fusion_info = generate_one(
                ex=ex,
                args=args,
                llm=llm,
                tokenizer=tokenizer,
                dlm=dlm,
                projector=projector,
                attention_fusion=attention_fusion,
                device=device
            )

            ex[output_key] = summary

            if fusion_info.get("fusion_weights") is not None:
                ex[f"{output_key}_weights"] = fusion_info["fusion_weights"]

            out.write(json.dumps(ex, ensure_ascii=False) + "\n")
            out.flush()

    print("Saved:", args.output)
    print("Output key:", output_key)


if __name__ == "__main__":
    main()