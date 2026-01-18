# make_output2_predictions.py
import argparse
import json
from typing import Any, Dict, Optional

from normalize_output2 import convert_output1_to_output2, FIELDS, DNA


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _try_parse_raw_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """
    Some runs pack multiple JSON objects / append text.
    Parse the first JSON object from the beginning.
    """
    raw = raw.strip()
    if not raw:
        return None

    # Fast path
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Slow path: bracket matching for first {...}
    if not raw.startswith("{"):
        i = raw.find("{")
        if i == -1:
            return None
        raw = raw[i:]

    depth = 0
    for i, ch in enumerate(raw):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = raw[: i + 1]
                try:
                    obj = json.loads(chunk)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    return None
    return None


def normalize_prediction_output_1(pred_out_1: Any) -> Dict[str, Any]:
    """
    Ensure prediction_output_1 is a dict with all expected fields.
    Handles {"__raw__": "..."} fallback.
    """
    if isinstance(pred_out_1, dict) and "__raw__" in pred_out_1:
        parsed = _try_parse_raw_json_object(str(pred_out_1["__raw__"]))
        if parsed is None:
            return {f: [DNA] for f in FIELDS}
        pred_out_1 = parsed

    if not isinstance(pred_out_1, dict):
        return {f: [DNA] for f in FIELDS}

    fixed = {}
    for f in FIELDS:
        fixed[f] = pred_out_1.get(f, [DNA])
    return fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="runs/.../predictions.jsonl (output_1)")
    ap.add_argument("--out", required=True, help="write converted predictions (output_2) jsonl")
    args = ap.parse_args()

    with open(args.out, "w", encoding="utf-8") as w:
        for row in load_jsonl(args.predictions):
            if row.get("_type") != "prediction":
                continue

            pred1 = normalize_prediction_output_1(row.get("prediction_output_1"))
            pred2 = convert_output1_to_output2(pred1)

            out_row = dict(row)
            out_row["prediction_output_2"] = pred2
            w.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
