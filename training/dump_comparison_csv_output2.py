# dump_comparison_csv_output2.py
import argparse
import csv
import json
import os
import re
from typing import Dict, List, Tuple

from normalize_output2 import FIELDS, DNA

def coerce_flat_str_list(vals):
    """
    Ensure vals becomes a flat List[str].
    Handles:
      - None
      - strings
      - list[str]
      - list[list[...]] (nested)
      - other scalars
    """
    out = []
    if vals is None:
        return [DNA]

    # wrap scalar
    if not isinstance(vals, list):
        vals = [vals]

    stack = list(vals)
    while stack:
        v = stack.pop(0)
        if v is None:
            continue
        if isinstance(v, list):
            stack = v + stack  # prepend to process in order
            continue
        # normalize to string
        s = str(v).strip()
        if not s:
            continue
        out.append(s)

    return out if out else [DNA]


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip()


def jaccard_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def partial_match(p: str, g: str, min_substring_chars: int = 6, jaccard_threshold: float = 0.45) -> bool:
    if not p or not g:
        return False
    pn = normalize_text(p)
    gn = normalize_text(g)

    # substring match
    if len(pn) >= min_substring_chars and pn in gn:
        return True
    if len(gn) >= min_substring_chars and gn in pn:
        return True

    # token jaccard match
    return jaccard_similarity(pn, gn) >= jaccard_threshold


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_gold_output2(test_path: str) -> List[Dict[str, List[str]]]:
    gold = []
    for row in load_jsonl(test_path):
        obj2 = json.loads(row["output_2"])
        fixed = {f: obj2.get(f, [DNA]) for f in FIELDS}
        gold.append(fixed)
    return gold


def load_pred_output2(pred_path: str) -> List[Tuple[str, Dict[str, List[str]]]]:
    preds = []
    for row in load_jsonl(pred_path):
        if row.get("_type") != "prediction":
            continue
        doc_id = row.get("document_id")
        obj2 = row.get("prediction_output_2", {})
        if not isinstance(obj2, dict):
            obj2 = {}
        fixed = {f: obj2.get(f, [DNA]) for f in FIELDS}
        preds.append((doc_id, fixed))
    return preds


def best_jaccard(pred_vals: List[str], gold_vals: List[str]) -> float:
    best = 0.0
    for p in pred_vals:
        for g in gold_vals:
            best = max(best, jaccard_similarity(normalize_text(p), normalize_text(g)))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions_output2", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("== dump_comparison_csv_output2 ==")
    print("predictions_output2:", args.predictions_output2)
    print("test:", args.test)
    print("out:", args.out)

    if not os.path.exists(args.predictions_output2):
        raise FileNotFoundError(f"Predictions file not found: {args.predictions_output2}")
    if not os.path.exists(args.test):
        raise FileNotFoundError(f"Test file not found: {args.test}")

    gold_rows = load_gold_output2(args.test)
    pred_rows = load_pred_output2(args.predictions_output2)

    print("gold rows:", len(gold_rows))
    print("pred rows:", len(pred_rows))

    if len(pred_rows) == 0:
        raise RuntimeError("No prediction rows found (_type='prediction'). Is this the right file?")
    if len(gold_rows) != len(pred_rows):
        raise RuntimeError(f"Row count mismatch: gold={len(gold_rows)} pred={len(pred_rows)}")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.abspath(args.out)

    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "document_id",
            "field",
            "gold_values",
            "pred_values",
            "exact_match",
            "partial_match",
            "best_jaccard",
        ])

        for (doc_id, pred), gold in zip(pred_rows, gold_rows):
            for field in FIELDS:
                g_vals = coerce_flat_str_list(gold.get(field, [DNA]))
                p_vals = coerce_flat_str_list(pred.get(field, [DNA]))

                # keep empty as DNA-like
                g_vals = [v if v else DNA for v in g_vals]
                p_vals = [v if v else DNA for v in p_vals]

                exact = set(p_vals) == set(g_vals)

                part = any(partial_match(p, g) for p in p_vals for g in g_vals)
                jac = best_jaccard(p_vals, g_vals)

                writer.writerow([
                    doc_id,
                    field,
                    " | ".join(g_vals),
                    " | ".join(p_vals),
                    int(exact),
                    int(part),
                    round(jac, 4),
                ])

        csvfile.flush()
        os.fsync(csvfile.fileno())

    print("✅ wrote:", out_path)
    print("✅ bytes:", os.path.getsize(out_path))


if __name__ == "__main__":
    main()
