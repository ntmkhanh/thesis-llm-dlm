import json, evaluate
from bert_score import score

rouge = evaluate.load("rouge")

def load_preds(path, pred_key):
    preds, refs = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            preds.append(ex[pred_key])
            refs.append(ex["reference"])
    return preds, refs

path = "data/test_drafts_50.jsonl"
pred_key = "draft"   # đổi thành "s1" nếu đánh giá S1

preds, refs = load_preds(path, pred_key)

r = rouge.compute(predictions=preds, references=refs)

P, R, F1 = score(preds, refs, lang="en", verbose=True)

print({
    "rouge1": r["rouge1"],
    "rouge2": r["rouge2"],
    "rougeL": r["rougeL"],
    "rougeLsum": r["rougeLsum"],
    "bertscore_precision": P.mean().item(),
    "bertscore_recall": R.mean().item(),
    "bertscore_f1": F1.mean().item()
})