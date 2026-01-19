import os, json, argparse
import numpy as np
from sentence_transformers import SentenceTransformer

def l2_normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", default="data/normalised_data.jsonl")
    ap.add_argument("--out_dir", default="data/emb_out")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max_chars", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--id_key", default="document_id")          # change if needed
    ap.add_argument("--text_key", default="context_text")       # change if needed
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model = SentenceTransformer(args.model)

    doc_ids = []
    texts = []

    n = 0
    for obj in iter_jsonl(args.in_jsonl):
        if args.limit and n >= args.limit:
            break

        # ---- CHANGE THESE TWO KEYS IF YOUR JSONL USES DIFFERENT NAMES ----
        did = obj.get(args.id_key)
        txt = obj.get(args.text_key, "") or ""
        # -----------------------------------------------------------------

        if not did:
            continue
        doc_ids.append(str(did))
        texts.append(txt[: args.max_chars])
        n += 1

    if not doc_ids:
        raise SystemExit("No documents found (check id_key/text_key).")

    all_vecs = []
    for i in range(0, len(texts), args.batch):
        chunk = texts[i:i+args.batch]
        vecs = model.encode(chunk, batch_size=args.batch, convert_to_numpy=True, show_progress_bar=False)
        vecs = vecs.astype("float32")
        vecs = l2_normalize(vecs)
        all_vecs.append(vecs)
        print(f"embedded {min(i+args.batch, len(texts))}/{len(texts)}")

    X = np.vstack(all_vecs)
    meta = {"model": args.model, "dim": int(X.shape[1]), "count": int(X.shape[0])}

    np.savez_compressed(os.path.join(args.out_dir, "doc_embeddings.npz"), X=X)
    with open(os.path.join(args.out_dir, "doc_ids.jsonl"), "w", encoding="utf-8") as f:
        for did in doc_ids:
            f.write(json.dumps({"document_id": did}) + "\n")
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)

    print("Wrote:", args.out_dir, meta)

if __name__ == "__main__":
    main()
