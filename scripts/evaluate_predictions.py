from collections import defaultdict
from typing import List, Dict, Set


DNA = "data not available"


def normalize(values: List[str]) -> Set[str]:
    s = {v.strip() for v in values if v and v.strip()}
    if not s:
        return {DNA}
    if DNA in s:
        return {DNA}
    return s


def score_sets(gold: Set[str], pred: Set[str]):
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    return tp, fp, fn


def evaluate(
    gold_rows: List[Dict[str, List[str]]],
    pred_rows: List[Dict[str, List[str]]],
    fields: List[str],
):
    assert len(gold_rows) == len(pred_rows)

    field_stats = {
        f: {"tp": 0, "fp": 0, "fn": 0, "support": 0}
        for f in fields
    }

    for gold, pred in zip(gold_rows, pred_rows):
        for f in fields:
            g = normalize(gold[f])
            p = normalize(pred[f])

            tp, fp, fn = score_sets(g, p)
            field_stats[f]["tp"] += tp
            field_stats[f]["fp"] += fp
            field_stats[f]["fn"] += fn
            field_stats[f]["support"] += 1

    results = {}

    micro_tp = micro_fp = micro_fn = 0

    for f, s in field_stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        results[f] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": s["support"],
        }

    micro_precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )

    macro_f1 = sum(r["f1"] for r in results.values()) / len(results)

    return {
        "per_field": results,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }
