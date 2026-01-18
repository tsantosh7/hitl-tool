#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from validate_output_1 import FIELDS, validate_output
from evaluate_predictions import evaluate

DNA = "data not available"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_gold_from_finetune_row(row: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    finetune jsonl: {"messages":[..., {"role":"assistant","content":"{...}"}]}
    """
    msgs = row.get("messages", [])
    if not isinstance(msgs, list):
        raise ValueError("test row missing messages[]")

    for m in reversed(msgs):
        if m.get("role") == "assistant":
            content = m.get("content")
            if isinstance(content, dict):
                gold = content
            elif isinstance(content, str):
                gold = json.loads(content)
            else:
                raise ValueError("assistant content not str/dict")
            break
    else:
        raise ValueError("no assistant message in row")

    # enforce schema + fill missing fields
    out = {}
    for f in FIELDS:
        v = gold.get(f, [DNA])
        if not isinstance(v, list) or not v:
            v = [DNA]
        out[f] = [str(x) for x in v]
    validate_output(out)
    return out


def extract_pred_from_predictions_row(obj: Dict[str, Any]) -> Tuple[str, Dict[str, List[str]]]:
    if obj.get("_type") != "prediction":
        raise ValueError("not a prediction row")

    doc_id = str(obj.get("document_id"))
    pred = obj.get("prediction_output_1")
    if not isinstance(pred, dict):
        pred = {}

    out = {}
    for f in FIELDS:
        v = pred.get(f, [DNA])
        if not isinstance(v, list) or not v:
            v = [DNA]
        out[f] = [str(x) for x in v]
    validate_output(out)
    return doc_id, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="runs/.../predictions.jsonl")
    ap.add_argument("--test", default="data/test.finetune.jsonl")
    ap.add_argument("--out_dir", default=None, help="optional override; default = predictions file folder")
    ap.add_argument("--topn", type=int, default=10, help="show best/worst N fields")
    args = ap.parse_args()

    pred_path = Path(args.predictions)
    test_path = Path(args.test)

    out_dir = Path(args.out_dir) if args.out_dir else pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load gold
    test_rows = load_jsonl(test_path)
    gold_rows: List[Dict[str, List[str]]] = [extract_gold_from_finetune_row(r) for r in test_rows]

    # Load predictions (by document_id)
    pred_rows_raw = load_jsonl(pred_path)
    pred_map: Dict[str, Dict[str, List[str]]] = {}
    for obj in pred_rows_raw:
        if obj.get("_type") != "prediction":
            continue
        doc_id, out = extract_pred_from_predictions_row(obj)
        pred_map[doc_id] = out

    # Align: assume doc_ids are row_0..row_n-1 unless otherwise stored
    aligned_pred_rows: List[Dict[str, List[str]]] = []
    missing = 0
    for i in range(len(gold_rows)):
        doc_id = f"row_{i}"
        pred = pred_map.get(doc_id)
        if pred is None:
            missing += 1
            pred = {f: [DNA] for f in FIELDS}
        aligned_pred_rows.append(pred)

    results = evaluate(gold_rows, aligned_pred_rows, FIELDS)

    # Write artifacts
    (out_dir / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # per-field TSV for easy viewing
    lines = ["field\tprecision\trecall\tf1\tsupport"]
    for f, r in results["per_field"].items():
        lines.append(f"{f}\t{r['precision']:.4f}\t{r['recall']:.4f}\t{r['f1']:.4f}\t{r['support']}")
    (out_dir / "per_field.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Test rows: {len(gold_rows)}")
    print(f"Missing predictions: {missing}")
    print(f"Micro F1: {results['micro_f1']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")

    # show best/worst fields by F1
    per_field = results["per_field"]
    ranked = sorted(per_field.items(), key=lambda kv: kv[1]["f1"])
    topn = args.topn

    print(f"\n--- Worst {topn} fields by F1 ---")
    for f, r in ranked[:topn]:
        print(f"{f:24s}  f1={r['f1']:.4f}  p={r['precision']:.4f}  r={r['recall']:.4f}")

    print(f"\n--- Best {topn} fields by F1 ---")
    for f, r in ranked[-topn:][::-1]:
        print(f"{f:24s}  f1={r['f1']:.4f}  p={r['precision']:.4f}  r={r['recall']:.4f}")

    print(f"\nArtifacts written to: {out_dir}\n")


if __name__ == "__main__":
    main()
