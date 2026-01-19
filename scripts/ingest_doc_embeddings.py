import os, json, argparse
import numpy as np
from sqlalchemy import create_engine, text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--model_override", default="")
    args = ap.parse_args()

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise SystemExit("DATABASE_URL not set")

    meta = json.load(open(os.path.join(args.in_dir, "meta.json"), "r", encoding="utf-8"))
    model = args.model_override.strip() or meta["model"]
    dim = int(meta["dim"])

    X = np.load(os.path.join(args.in_dir, "doc_embeddings.npz"))["X"].astype("float32")
    ids = [json.loads(l)["document_id"] for l in open(os.path.join(args.in_dir, "doc_ids.jsonl"), "r", encoding="utf-8")]

    if len(ids) != X.shape[0]:
        raise SystemExit(f"IDs ({len(ids)}) != embeddings rows ({X.shape[0]})")

    engine = create_engine(db_url)

    upsert = text("""
    INSERT INTO doc_embeddings (document_id, embedding_dim, model, embedding, updated_at)
    VALUES (:document_id, :dim, :model, :emb, NOW())
    ON CONFLICT (document_id) DO UPDATE SET
      embedding_dim = EXCLUDED.embedding_dim,
      model = EXCLUDED.model,
      embedding = EXCLUDED.embedding,
      updated_at = NOW()
    """)

    total = 0
    with engine.begin() as conn:
        for i, did in enumerate(ids):
            conn.execute(upsert, {
                "document_id": did,
                "dim": dim,
                "model": model,
                "emb": X[i].tobytes(),
            })
            total += 1
            if total % 5000 == 0:
                print("upserted", total)

    print("DONE ingest embeddings:", total, "model=", model, "dim=", dim)

if __name__ == "__main__":
    main()
