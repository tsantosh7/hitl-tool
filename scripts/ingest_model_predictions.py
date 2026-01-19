#!/usr/bin/env python3
import os, json, argparse
from datetime import datetime

import psycopg


def coerce_db_url(db_url: str) -> str:
    # psycopg can't parse SQLAlchemy dialect URLs like postgresql+psycopg://
    if db_url.startswith("postgresql+psycopg://"):
        return "postgresql://" + db_url[len("postgresql+psycopg://"):]
    if db_url.startswith("postgresql+psycopg2://"):
        return "postgresql://" + db_url[len("postgresql+psycopg2://"):]
    return db_url

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="path to predictions.jsonl")
    ap.add_argument("--run_id_file", required=True, help="path to run_id.txt")
    ap.add_argument("--db_url", default=os.getenv("DATABASE_URL"), help="postgres url (or set DATABASE_URL)")
    ap.add_argument("--commit_every", type=int, default=500)
    args = ap.parse_args()

    if not args.db_url:
        raise SystemExit("Missing --db_url or DATABASE_URL")

    run_id = open(args.run_id_file, "r", encoding="utf-8").read().strip()

    create_sql = """
    CREATE TABLE IF NOT EXISTS model_predictions (
      document_id TEXT NOT NULL,
      run_id TEXT NOT NULL,
      model TEXT,
      k_shot INTEGER,
      seed INTEGER,
      prediction_output_1 JSONB,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      PRIMARY KEY (document_id, run_id)
    );
    CREATE INDEX IF NOT EXISTS idx_model_predictions_run_id ON model_predictions(run_id);
    """

    upsert_sql = """
    INSERT INTO model_predictions (document_id, run_id, model, k_shot, seed, prediction_output_1, updated_at)
    VALUES (%s, %s, %s, %s, %s, %s::jsonb, NOW())
    ON CONFLICT (document_id, run_id)
    DO UPDATE SET
      model = EXCLUDED.model,
      k_shot = EXCLUDED.k_shot,
      seed = EXCLUDED.seed,
      prediction_output_1 = EXCLUDED.prediction_output_1,
      updated_at = NOW();
    """

    meta = {}  # from _type=run_metadata if present
    n_pred = 0
    n_skipped = 0

    db_url = coerce_db_url(args.db_url)
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(create_sql)
            conn.commit()

            with open(args.pred, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    rtype = row.get("_type")

                    if rtype == "run_metadata":
                        # use this to enrich each prediction row
                        meta = row
                        continue

                    if rtype != "prediction":
                        n_skipped += 1
                        continue

                    if row.get("run_id") != run_id:
                        # safety: don't mix runs
                        n_skipped += 1
                        continue

                    did = row.get("document_id")
                    out1 = row.get("prediction_output_1")
                    if not did or not isinstance(out1, dict):
                        n_skipped += 1
                        continue

                    model = meta.get("model") or row.get("model")
                    k_shot = meta.get("k_shot") or meta.get("k_shot".replace("_", ""))  # just in case
                    if k_shot is None:
                        k_shot = meta.get("k_shot") or meta.get("k_shot".replace("_", ""))  # no-op safety
                    # your metadata uses "k_shot" key
                    k_shot = meta.get("k_shot") if meta else None
                    seed = meta.get("seed") if meta else None

                    cur.execute(upsert_sql, (did, run_id, model, k_shot, seed, json.dumps(out1)))
                    n_pred += 1

                    if n_pred % args.commit_every == 0:
                        conn.commit()
                        print(f"[{datetime.utcnow().isoformat()}Z] upserted {n_pred} predictions...")

            conn.commit()

    print(f"Done. Upserted {n_pred} predictions. Skipped {n_skipped} rows.")

if __name__ == "__main__":
    main()
