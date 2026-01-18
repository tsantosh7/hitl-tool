#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Tuple

DNA = "data not available"

# Import schema field list (authoritative)
from validate_output_1 import FIELDS  # noqa


# -------------------------
# Normalization utilities
# -------------------------
_ws = re.compile(r"\s+")
_punct_tbl = str.maketrans("", "", string.punctuation)

def norm_basic(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    s = _ws.sub(" ", s).strip().casefold()
    return s

def norm_loose(s: str) -> str:
    s = norm_basic(s)
    s = s.translate(_punct_tbl)
    s = _ws.sub(" ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    s = norm_loose(s)
    return s.split() if s else []

def is_dna(x: str) -> bool:
    return norm_basic(x) == DNA

# -------------------------
# Similarity metrics
# -------------------------
def jaccard(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def substring_match(a: str, b: str, min_chars: int = 6) -> bool:
    a_n = norm_basic(a)
    b_n = norm_basic(b)
    if len(a_n) >= min_chars and a_n in b_n:
        return True
    if len(b_n) >= min_chars and b_n in a_n:
        return True
    return False

def exact_match(a: str, b: str) -> bool:
    if is_dna(a) and is_dna(b):
        return True
    if is_dna(a) or is_dna(b):
        return False
    return norm_loose(a) == norm_loose(b)

# -------------------------
# Loading helpers
# -------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def safe_list(x: Any) -> List[str]:
    if isinstance(x, list) and x:
        return [str(v) for v in x]
    return [DNA]

def extract_gold_from_finetune_row(row: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    finetune jsonl row: {"messages":[..., {"role":"assistant","content":"{...}"}]}
    Returns dict[field] -> list[str]
    """
    msgs = row.get("messages")
    if not isinstance(msgs, list):
        raise ValueError("Expected 'messages' list in finetune row")

    gold_obj = None
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            content = m.get("content")
            if isinstance(content, dict):
                gold_obj = content
            elif isinstance(content, str):
                gold_obj = json.loads(content)
            else:
                raise ValueError("assistant content must be dict or json string")
            break
    if gold_obj is None:
        raise ValueError("No assistant message found in finetune row")

    out: Dict[str, List[str]] = {}
    for f in FIELDS:
        out[f] = safe_list(gold_obj.get(f))
    return out

# -------------------------
# Pair selection per field
# -------------------------
def best_pair(pred_vals: List[str], gold_vals: List[str]) -> Tuple[str, str, float, bool, bool]:
    """
    Choose the (pred_item, gold_item) pair with maximum Jaccard.
    Returns:
      pred_item, gold_item, best_jaccard, best_is_substring, best_is_exact
    """
    best_j = -1.0
    best_p = pred_vals[0] if pred_vals else DNA
    best_g = gold_vals[0] if gold_vals else DNA
    best_sub = False
    best_ex = False

    for g in gold_vals:
        for p in pred_vals:
            j = jaccard(p, g)
            if j > best_j:
                best_j = j
                best_p = p
                best_g = g
                best_sub = substring_match(p, g)
                best_ex = exact_match(p, g)

    # clamp
    if best_j < 0:
        best_j = 0.0
    return best_p, best_g, round(best_j, 4), best_sub, best_ex

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="runs/.../predictions.jsonl")
    ap.add_argument("--test", required=True, help="data/test.finetune.jsonl")
    ap.add_argument("--out", default="comparisons.csv")
    ap.add_argument("--min_substring_chars", type=int, default=6)
    args = ap.parse_args()

    pred_rows_raw = load_jsonl(Path(args.predictions))
    test_rows = load_jsonl(Path(args.test))

    # index predictions by document_id
    pred_map: Dict[str, Dict[str, Any]] = {}
    for r in pred_rows_raw:
        if r.get("_type") != "prediction":
            continue
        doc_id = str(r.get("document_id"))
        pred_map[doc_id] = r.get("prediction_output_1", {}) or {}

    # write CSV: 1 row per (doc_id, field) with best-match similarity
    out_path = Path(args.out)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "document_id",
                "field",
                "gold_items",
                "pred_items",
                "best_gold_item",
                "best_pred_item",
                "jaccard_best",
                "substring_best",
                "exact_best",
                "gold_is_dna_only",
                "pred_is_dna_only",
            ],
        )
        w.writeheader()

        for i, test_row in enumerate(test_rows):
            doc_id = f"row_{i}"
            gold = extract_gold_from_finetune_row(test_row)
            pred = pred_map.get(doc_id, {})

            for field in FIELDS:
                gold_vals = safe_list(gold.get(field))
                pred_vals = safe_list(pred.get(field))

                # best-pair similarity for quick filtering
                bp, bg, bj, bsub, bex = best_pair(pred_vals, gold_vals)

                w.writerow({
                    "document_id": doc_id,
                    "field": field,
                    "gold_items": json.dumps(gold_vals, ensure_ascii=False),
                    "pred_items": json.dumps(pred_vals, ensure_ascii=False),
                    "best_gold_item": bg,
                    "best_pred_item": bp,
                    "jaccard_best": bj,
                    "substring_best": bsub,
                    "exact_best": bex,
                    "gold_is_dna_only": (len(gold_vals) == 1 and is_dna(gold_vals[0])),
                    "pred_is_dna_only": (len(pred_vals) == 1 and is_dna(pred_vals[0])),
                })

    print(f"✅ Wrote CSV with Jaccard to: {out_path}")

if __name__ == "__main__":
    main()

