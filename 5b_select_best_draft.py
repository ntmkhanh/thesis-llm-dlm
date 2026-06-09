import json, re, argparse, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/test_multi_drafts_500.jsonl")
    p.add_argument("--output", default="data/test_best_draft_500.jsonl")
    p.add_argument("--nli_model", default="roberta-large-mnli")
    return p.parse_args()


def split_sentences(text):
    text = text.replace("\n", " ").strip()
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if len(s.strip()) > 5]


class SummaCZS:
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.entail_idx = 2

    @torch.no_grad()
    def pair_scores(self, premises, hypotheses):
        inputs = self.tok(
            premises,
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        probs = torch.softmax(self.model(**inputs).logits, dim=-1)
        return probs[:, self.entail_idx].detach().cpu().tolist()

    def score(self, article, summary):
        doc_sents = split_sentences(article)
        sum_sents = split_sentences(summary)

        if not doc_sents or not sum_sents:
            return 0.0

        scores = []
        for s in sum_sents:
            premises = doc_sents
            hypotheses = [s] * len(doc_sents)
            scores.append(max(self.pair_scores(premises, hypotheses)))

        return sum(scores) / len(scores)


def main():
    args = parse_args()
    scorer = SummaCZS(args.nli_model)

    with open(args.input, encoding="utf-8") as f, open(args.output, "w", encoding="utf-8") as out:
        for line in tqdm(f, desc="Selecting best draft"):
            ex = json.loads(line)

            scored = []
            for d in tqdm(
                ex["drafts"],
                leave=False,
                desc="Scoring drafts"
            ):

                scored.append((scorer.score(ex["article"], d), d))

            scored.sort(reverse=True, key=lambda x: x[0])

            ex["best_draft"] = scored[0][1]
            ex["best_draft_summac"] = scored[0][0]
            ex["draft_scores"] = [s for s, _ in scored]

            out.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("Saved:", args.output)


if __name__ == "__main__":
    main()
