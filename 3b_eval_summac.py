import json
import re
import torch
import evaluate
from tqdm import tqdm
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =========================
# CONFIG
# =========================

FILE_PATH = "data/test_pp2_true_diffusion.jsonl"
PRED_KEY = "pp2_summary"

# Với PP1:
# FILE_PATH = "data/test_s1.jsonl"
# PRED_KEY = "s1"

# Với baseline:
# FILE_PATH = "data/test_drafts.jsonl"
# PRED_KEY = "draft"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NLI_MODEL = "roberta-large-mnli"
BATCH_SIZE = 16


# =========================
# LOAD DATA
# =========================

def load_jsonl(path, pred_key):
    articles, preds, refs = [], [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)

            if pred_key not in ex:
                continue

            articles.append(ex["article"])
            preds.append(ex[pred_key])
            refs.append(ex["reference"])

    return articles, preds, refs


# =========================
# SIMPLE SENTENCE SPLIT
# =========================

def split_sentences(text):
    text = text.replace("\n", " ").strip()
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in sents if len(s.strip()) > 5]
    return sents


# =========================
# SUMMAC-ZS
# =========================

class SummaCZS:
    def __init__(self, model_name=NLI_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(DEVICE)
        self.model.eval()

        # roberta-large-mnli labels:
        # 0 = contradiction, 1 = neutral, 2 = entailment
        self.entailment_idx = 2

    @torch.no_grad()
    def entailment_scores(self, premises, hypotheses):
        scores = []

        for i in range(0, len(premises), BATCH_SIZE):
            p_batch = premises[i:i + BATCH_SIZE]
            h_batch = hypotheses[i:i + BATCH_SIZE]

            inputs = self.tokenizer(
                p_batch,
                h_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(DEVICE)

            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

            entail = probs[:, self.entailment_idx]
            scores.extend(entail.detach().cpu().tolist())

        return scores

    def score_one(self, document, summary):
        doc_sents = split_sentences(document)
        sum_sents = split_sentences(summary)

        if len(doc_sents) == 0 or len(sum_sents) == 0:
            return 0.0

        summary_scores = []

        for s in sum_sents:
            premises = doc_sents
            hypotheses = [s] * len(doc_sents)

            entail_scores = self.entailment_scores(premises, hypotheses)

            # SUMMAC-ZS: max theo document sentences
            best_support = max(entail_scores)
            summary_scores.append(best_support)

        # mean theo summary sentences
        return sum(summary_scores) / len(summary_scores)

    def score_batch(self, documents, summaries):
        results = []

        for doc, summ in tqdm(
            zip(documents, summaries),
            total=len(documents),
            desc="Computing SUMMAC-ZS"
        ):
            results.append(self.score_one(doc, summ))

        return results


# =========================
# MAIN EVAL
# =========================

articles, preds, refs = load_jsonl(FILE_PATH, PRED_KEY)

print("n_samples:", len(preds))

# ROUGE
rouge = evaluate.load("rouge")
rouge_result = rouge.compute(
    predictions=preds,
    references=refs
)

# BERTScore
P, R, F1 = bert_score(
    preds,
    refs,
    lang="en",
    verbose=True
)

# SUMMAC-ZS
summac = SummaCZS()
summac_scores = summac.score_batch(articles, preds)
avg_summac = sum(summac_scores) / len(summac_scores)

result = {
    "n_samples": len(preds),

    "rouge1": rouge_result["rouge1"],
    "rouge2": rouge_result["rouge2"],
    "rougeL": rouge_result["rougeL"],
    "rougeLsum": rouge_result["rougeLsum"],

    "bertscore_precision": P.mean().item(),
    "bertscore_recall": R.mean().item(),
    "bertscore_f1": F1.mean().item(),

    "summac_zs": avg_summac
}

print(json.dumps(result, indent=2))