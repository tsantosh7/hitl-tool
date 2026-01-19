#!/usr/bin/env python3
import os
import json
import argparse
from typing import Dict, Any, List, Tuple

import psycopg
import requests

import sys
if "/app" not in sys.path:
    sys.path.insert(0, "/app")

from training.normalize_output2 import convert_output1_to_output2 as normalize_output2



SENTINEL_DEFAULT = "data not available"



def build_model_fields(out1: Dict[str, Any], sentinel: str) -> Tuple[List[str], List[str], str]:
    codes_present: List[str] = []
    kv: List[str] = []
    lines: List[str] = []

    for code, vals in (out1 or {}).items():
        if not isinstance(code, str):
            continue
        if vals is None:
            continue
        if isinstance(vals, str):
            vals = [vals]
        if not isinstance(vals, list):
            continue

        clean_vals = []
        for v in vals:
            if not isinstance(v, str):
                continue
            vv = v.strip()
            if not vv or vv.lower() == sentinel.lower():
                continue
            clean_vals.append(vv)

        if not clean_vals:
            continue

        codes_present.append(code)
        for vv in clean_vals:
            kv.append(f"{code}={vv}")
            lines.append(f"{code}: {vv}")

    return codes_present, kv, "\n".join(lines)


def coerce_db_url(db_url: str) -> str:
    # psycopg cannot parse SQLAlchemy-style URLs like postgresql+psycopg://
    if db_url.startswith("postgresql+psycopg://"):
        return "postgresql://" + db_url[len("postgresql+psycopg://"):]
    if db_url.startswith("postgresql+psycopg2://"):
        return "postgresql://" + db_url[len("postgresql+psycopg2://"):]
    return db_url


def build_model_fields(out1: Dict[str, Any], sentinel: str) -> Tuple[List[str], List[str], str]:
    """
    From prediction_output_1 (field -> list[str]),
    build:
      - codes_present_model_ss (list of code names with any non-sentinel value)
      - code_value_model_kv_ss (["Code=value", ...]) for exact filtering
      - values_model_txt (big free-text blob for searching)
    """
    codes_present: List[str] = []
    kv: List[str] = []
    lines: List[str] = []

    for code, vals in out1.items():
        if not isinstance(code, str):
            continue
        if vals is None:
            continue
        if isinstance(vals, str):
            vals = [vals]
        if not isinstance(vals, list):
            continue

        clean_vals = []
        for v in vals:
            if not isinstance(v, str):
                continue
            vv = v.strip()
            if not vv or vv.lower() == sentinel.lower():
                continue
            clean_vals.append(vv)

        if not clean_vals:
            continue

        codes_present.append(code)
        for vv in clean_vals:
            kv.append(f"{code}={vv}")
            lines.append(f"{code}: {vv}")

    values_blob = "\n".join(lines)
    return codes_present, kv, values_blob


def chunked(items: List[Dict[str, Any]], n: int):
    for i in range(0, len(items), n):
        yield items[i:i+n]


def solr_atomic_update(solr_base: str, core: str, docs: List[Dict[str, Any]], timeout: int = 120) -> None:
    url = f"{solr_base.rstrip('/')}/{core}/update"
    params = {"commitWithin": "10000"}  # 10s commitWithin, avoids huge commit storms
    r = requests.post(url, params=params, json=docs, timeout=timeout)
    r.raise_for_status()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default="hitl_test")
    ap.add_argument("--solr_base", default=os.getenv("SOLR_BASE_URL", "http://localhost:8983/solr"))
    ap.add_argument("--db_url", default=os.getenv("DATABASE_URL"))
    ap.add_argument("--run_id_file", required=True)
    ap.add_argument("--sentinel", default=SENTINEL_DEFAULT)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    if not args.db_url:
        raise SystemExit("Missing --db_url or DATABASE_URL")

    run_id = open(args.run_id_file, "r", encoding="utf-8").read().strip()
    db_url = coerce_db_url(args.db_url)

    # Pull model predictions
    sql = """
      SELECT document_id, prediction_output_1
      FROM model_predictions
      WHERE run_id = %s
      ORDER BY document_id
    """

    updates: List[Dict[str, Any]] = []
    n = 0

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (run_id,))
            for (document_id, out1) in cur:
                if out1 is None:
                    continue
                if not isinstance(out1, dict):
                    # psycopg returns JSONB as Python objects; should be dict
                    continue

                codes_present, kv, values_blob = build_model_fields(out1, args.sentinel)

                out2 = normalize_output2(out1)  # dict[str, list[str]] expected
                codes_present_n, kv_n, values_blob_n = build_model_fields(out2, args.sentinel)

                doc = {
                    "document_id_s": document_id,
                    "has_model_b": {"set": True},
                    "codes_present_model_ss": {"set": codes_present},
                    "code_value_model_kv_ss": {"set": kv},
                    "values_model_txt": {"set": values_blob},

                    # NEW: normalised value search + KV
                    "code_value_model_norm_kv_ss": {"set": kv_n},
                    "values_model_norm_txt": {"set": values_blob_n},
                }

                updates.append(doc)
                n += 1

                if args.limit and n >= args.limit:
                    break

    if not updates:
        print("No model predictions found for run_id:", run_id)
        return

    # Push to Solr in chunks
    pushed = 0
    for batch in chunked(updates, args.chunk):
        solr_atomic_update(args.solr_base, args.core, batch)
        pushed += len(batch)
        print(f"Pushed {pushed}/{len(updates)} docs to Solr...")

    print(f"Done. Updated {pushed} docs in Solr core={args.core} for run_id={run_id}.")


if __name__ == "__main__":
    main()
