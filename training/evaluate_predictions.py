# training/evaluate_predictions.py
from __future__ import annotations

import re
import string
from typing import Dict, List, Any, Tuple, Callable

DNA = "data not available"


# -----------------------------
# FIELD MATCH POLICY
# -----------------------------

EXACT_TAGS = [

]

PARTIAL_TAGS = [
    "ConvCourtName",
    "ConfessPleadGuilty",  # (your schema uses this key)
    "SentCourtName",
    "SentServe",
    "OffSex",
    "OffAgeOffence",
    "OffIntoxOffence",
    "VictimType",
    "VicNum",
    "VicSex",
    "VicAgeOffence",
    "VicIntoxOffence",
    "PreSentReport",
    "VicImpactStatement",
    "Appellant",
    "AppealAgainst",
    "AppealOutcome", # last exact here
    "ConvictPleaDate",
    "ConvictOffence",
    "AcquitOffence",
    "PleaPoint",
    "RemandDecision",
    "RemandCustodyTime",
    "Sentence",
    "WhatAncillary",
    "OffJobOffence",
    "OffHomeOffence",
    "OffMentalOffence",
    "OffVicRelation",
    "VicJobOffence",
    "VicHomeOffence",
    "VicMentalOffence",
    "ProsEvidTypeTrial",
    "DefEvidTypeTrial",
    "AggFactSent",
    "MitFactSent",
    "CoDefAccNum",
    "SentGuideWhich",
    "AppealGround",
    "ReasonQuashConv",
    "ReasonSentExcessNotLenient",
    "ReasonSentLenientNotExcess",
    "ReasonDismiss",
]

# soft-match thresholds
DEFAULT_PARTIAL_SUBSTRING_MIN_CHARS = 8
DEFAULT_PARTIAL_JACCARD_THRESHOLD = 0.70

PARTIAL_FIELD_PARAMS = {
    # long free-text / reasons / factors: allow looser matching
    "AppealGround": dict(min_substring_chars=5, jaccard_threshold=0.35),
    "AggFactSent": dict(min_substring_chars=5, jaccard_threshold=0.35),
    "MitFactSent": dict(min_substring_chars=5, jaccard_threshold=0.35),
    "ReasonDismiss": dict(min_substring_chars=5, jaccard_threshold=0.35),
    "Sentence": dict(min_substring_chars=6, jaccard_threshold=0.40),

    # dates/offences/courts: keep tighter (less risk of accidental overlap)
    "ConvictPleaDate": dict(min_substring_chars=8, jaccard_threshold=0.55),
    "ConvictOffence": dict(min_substring_chars=8, jaccard_threshold=0.55),
    "SentGuideWhich": dict(min_substring_chars=10, jaccard_threshold=0.50),
}


# -----------------------------
# NORMALIZATION
# -----------------------------

_ws_re = re.compile(r"\s+")
_punct_tbl = str.maketrans("", "", string.punctuation)

def norm_ws(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    s = _ws_re.sub(" ", s).strip()
    return s

def norm_basic(s: str) -> str:
    # casefold + whitespace normalize
    return norm_ws(s).casefold()

def norm_loose(s: str) -> str:
    # remove punctuation too (for exact tags)
    s = norm_basic(s)
    s = s.translate(_punct_tbl)
    s = _ws_re.sub(" ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    s = norm_loose(s)
    if not s:
        return []
    return [t for t in s.split(" ") if t]


def is_dna(x: str) -> bool:
    return norm_basic(x) == DNA


# -----------------------------
# MATCH FUNCTIONS
# -----------------------------

def exact_match(a: str, b: str) -> bool:
    if is_dna(a) and is_dna(b):
        return True
    if is_dna(a) or is_dna(b):
        return False
    return norm_loose(a) == norm_loose(b)

def partial_match(
    pred: str,
    gold: str,
    *,
    min_substring_chars: int = DEFAULT_PARTIAL_SUBSTRING_MIN_CHARS,
    jaccard_threshold: float = DEFAULT_PARTIAL_JACCARD_THRESHOLD,
) -> bool:
    """
    Soft match for verbatim spans.

    True if:
    - either is DNA and both DNA -> True; else if one DNA -> False
    - OR normalized substring containment (either direction) with a minimum length
    - OR token Jaccard similarity >= threshold
    """
    if is_dna(pred) and is_dna(gold):
        return True
    if is_dna(pred) or is_dna(gold):
        return False

    p = norm_basic(pred)
    g = norm_basic(gold)

    # substring containment (either direction)
    if len(p) >= min_substring_chars and p in g:
        return True
    if len(g) >= min_substring_chars and g in p:
        return True

    # token overlap (Jaccard)
    pt = set(tokenize(pred))
    gt = set(tokenize(gold))
    if not pt or not gt:
        return False
    j = len(pt & gt) / max(1, len(pt | gt))
    return j >= jaccard_threshold


def pick_matcher(field: str) -> Callable[[str, str], bool]:
    if field in EXACT_TAGS:
        return exact_match
    if field in PARTIAL_TAGS:
        params = PARTIAL_FIELD_PARAMS.get(field, {})
        return lambda p, g: partial_match(p, g, **params)
    # default: be conservative (exact)
    return exact_match


# -----------------------------
# LIST-LEVEL SCORING
# -----------------------------

def best_bipartite_matches(
    pred_list: List[str],
    gold_list: List[str],
    matcher: Callable[[str, str], bool],
) -> Tuple[int, int, int]:
    """
    Returns (tp, fp, fn) for list-valued fields using greedy matching.
    Greedy is fine here because lists are short; you can upgrade to Hungarian later.

    We match each pred to at most one gold.
    """
    preds = [p for p in pred_list if isinstance(p, str) and p.strip()]
    golds = [g for g in gold_list if isinstance(g, str) and g.strip()]

    used_gold = [False] * len(golds)
    tp = 0

    for p in preds:
        matched = False
        for j, g in enumerate(golds):
            if used_gold[j]:
                continue
            if matcher(p, g):
                used_gold[j] = True
                tp += 1
                matched = True
                break
        # if no match, it's a FP
    fp = len(preds) - tp
    fn = len(golds) - tp
    return tp, fp, fn


def safe_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(v) for v in x]
    return [DNA]


# -----------------------------
# PUBLIC API
# -----------------------------

def evaluate(
    gold_rows: List[Dict[str, Any]],
    pred_rows: List[Dict[str, Any]],
    fields: List[str],
) -> Dict[str, Any]:
    """
    Returns:
      {
        micro_f1, macro_f1,
        per_field: { field: {precision, recall, f1, support} },
        micro_f1_exact, micro_f1_relaxed,
      }

    NOTE:
    - "support" = total gold items across dataset for that field (list items)
    """
    per_field: Dict[str, Dict[str, Any]] = {}

    total_tp = total_fp = total_fn = 0
    total_tp_exact = total_fp_exact = total_fn_exact = 0
    total_tp_relaxed = total_fp_relaxed = total_fn_relaxed = 0

    for field in fields:
        matcher = pick_matcher(field)
        is_relaxed = (field in PARTIAL_TAGS)

        tp = fp = fn = 0
        tp_e = fp_e = fn_e = 0
        tp_r = fp_r = fn_r = 0

        support = 0

        for gold, pred in zip(gold_rows, pred_rows):
            g_list = safe_list(gold.get(field))
            p_list = safe_list(pred.get(field))

            support += len(g_list)

            # field policy scoring
            a, b, c = best_bipartite_matches(p_list, g_list, matcher)
            tp += a; fp += b; fn += c

            # strict exact scoring (always exact_match)
            ae, be, ce = best_bipartite_matches(p_list, g_list, exact_match)
            tp_e += ae; fp_e += be; fn_e += ce

            # relaxed scoring: for PARTIAL_TAGS use partial_match, else exact_match
            if is_relaxed:
                params = PARTIAL_FIELD_PARAMS.get(field, {})
                ar, br, cr = best_bipartite_matches(p_list, g_list, lambda p, g: partial_match(p, g, **params))
            else:
                ar, br, cr = best_bipartite_matches(p_list, g_list, exact_match)
            tp_r += ar; fp_r += br; fn_r += cr

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        per_field[field] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "policy": ("partial" if field in PARTIAL_TAGS else "exact"),
        }

        total_tp += tp; total_fp += fp; total_fn += fn
        total_tp_exact += tp_e; total_fp_exact += fp_e; total_fn_exact += fn_e
        total_tp_relaxed += tp_r; total_fp_relaxed += fp_r; total_fn_relaxed += fn_r

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0

    # macro over fields
    macro_f1 = sum(per_field[f]["f1"] for f in fields) / max(1, len(fields))

    # strict micro
    mp_e = total_tp_exact / (total_tp_exact + total_fp_exact) if (total_tp_exact + total_fp_exact) else 0.0
    mr_e = total_tp_exact / (total_tp_exact + total_fn_exact) if (total_tp_exact + total_fn_exact) else 0.0
    micro_f1_exact = (2 * mp_e * mr_e / (mp_e + mr_e)) if (mp_e + mr_e) else 0.0

    # relaxed micro
    mp_r = total_tp_relaxed / (total_tp_relaxed + total_fp_relaxed) if (total_tp_relaxed + total_fp_relaxed) else 0.0
    mr_r = total_tp_relaxed / (total_tp_relaxed + total_fn_relaxed) if (total_tp_relaxed + total_fn_relaxed) else 0.0
    micro_f1_relaxed = (2 * mp_r * mr_r / (mp_r + mr_r)) if (mp_r + mr_r) else 0.0

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "micro_f1_exact": micro_f1_exact,
        "micro_f1_relaxed": micro_f1_relaxed,
        "per_field": per_field,
    }
