# score_output2_predictions.py
import argparse
import json
from typing import Dict, List

from evaluate_predictions import evaluate
from normalize_output2 import FIELDS, DNA


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_gold_output2(test_path: str) -> List[Dict[str, List[str]]]:
    """
    test.jsonl has: {"context": "...", "output_1": "<json string>", "output_2": "<json string>"}
    """
    gold = []
    for row in load_jsonl(test_path):
        obj2 = json.loads(row["output_2"])
        fixed = {f: obj2.get(f, [DNA]) for f in FIELDS}
        gold.append(fixed)
    return gold


def load_pred_output2(pred_path: str) -> List[Dict[str, List[str]]]:
    """
    predictions.output2.jsonl has rows with prediction_output_2 already materialized.
    Must align 1:1 with test rows.
    """
    preds = []
    for row in load_jsonl(pred_path):
        if row.get("_type") != "prediction":
            continue
        obj2 = row.get("prediction_output_2")
        if not isinstance(obj2, dict):
            obj2 = {}
        fixed = {f: obj2.get(f, [DNA]) for f in FIELDS}
        preds.append(fixed)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions_output2", required=True, help="output of make_output2_predictions.py")
    ap.add_argument("--test", required=True, help="data/test.jsonl (gold output_2)")
    ap.add_argument("--out_json", default=None, help="write evaluation JSON")
    args = ap.parse_args()

    gold_rows = load_gold_output2(args.test)
    pred_rows = load_pred_output2(args.predictions_output2)

    if len(gold_rows) != len(pred_rows):
        raise SystemExit(f"Row count mismatch: gold={len(gold_rows)} pred={len(pred_rows)}")

    results = evaluate(gold_rows, pred_rows, FIELDS)

    print("\n=== OUTPUT_2 EVALUATION SUMMARY ===")
    print(f"Test rows: {len(gold_rows)}")
    print(f"Micro F1: {results['micro_f1']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")

    per = results["per_field"]
    worst = sorted(per.items(), key=lambda kv: kv[1]["f1"])[:10]
    best = sorted(per.items(), key=lambda kv: kv[1]["f1"], reverse=True)[:10]

    print("\n--- Worst 10 fields by F1 ---")
    for f, r in worst:
        print(f"{f:24s} f1={r['f1']:.4f}  p={r['precision']:.4f}  r={r['recall']:.4f}")

    print("\n--- Best 10 fields by F1 ---")
    for f, r in best:
        print(f"{f:24s} f1={r['f1']:.4f}  p={r['precision']:.4f}  r={r['recall']:.4f}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as w:
            json.dump(results, w, indent=2, ensure_ascii=False)
        print(f"\nWrote: {args.out_json}")


if __name__ == "__main__":
    main()
