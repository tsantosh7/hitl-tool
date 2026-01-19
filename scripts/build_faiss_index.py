import os
import json
import argparse

import numpy as np
from sqlalchemy import create_engine, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    ap.add_argument("--faiss_dir", default=os.getenv("FAISS_DIR", "/data/faiss"))
    args = ap.parse_args()

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise SystemExit("DATABASE_URL not set in container env")

    os.makedirs(args.faiss_dir, exist_ok=True)
    index_path = os.path.join(args.faiss_dir, "docs.index")
    ids_path = os.path.join(args.faiss_dir, "docs_ids.jsonl")

    engine = create_engine(db_url)

    # Load embeddings in stable order
    sql = text("""
    SELECT document_id, embedding_dim, embedding
    FROM doc_embeddings
    WHERE model = :model
    ORDER BY document_id
    """)

    doc_ids = []
    vecs = []
    dim = None

    with engine.connect() as conn:
        for did, d, b in conn.execute(sql, {"model": args.model}):
            if dim is None:
                dim = int(d)
            if int(d) != dim:
                continue
            v = np.frombuffer(b, dtype=np.float32)
            if v.shape[0] != dim:
                continue
            doc_ids.append(did)
            vecs.append(v)

    if not vecs:
        raise SystemExit("No embeddings found for this model. Run build_doc_embeddings.py first.")

    X = np.stack(vecs, axis=0).astype("float32")

    import faiss  # faiss-cpu

    # Cosine similarity = inner product if vectors are L2-normalized
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    faiss.write_index(index, index_path)

    with open(ids_path, "w", encoding="utf-8") as f:
        for i, did in enumerate(doc_ids):
            f.write(json.dumps({"i": i, "document_id": did}) + "\n")

    print(f"Wrote FAISS index: {index_path}")
    print(f"Wrote id map:    {ids_path}")
    print(f"Docs indexed:    {len(doc_ids)}  dim={dim}")


if __name__ == "__main__":
    main()
