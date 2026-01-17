#!/usr/bin/env python3
import argparse
import json
import sys
import time
from typing import Iterable, List, Dict, Any, Optional

import requests


def chunks(iterable: Iterable[dict], n: int) -> Iterable[List[dict]]:
    buf: List[dict] = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def iter_jsonl(path: str, resume_from: int = 0, max_docs: int = 0) -> Iterable[dict]:
    """
    resume_from: skip first N records
    max_docs: stop after N yielded records (0 = unlimited)
    """
    yielded = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < resume_from:
                continue
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            yield obj

            yielded += 1
            if max_docs and yielded >= max_docs:
                break


def solr_hard_commit(solr_base: str, core: str, timeout: int = 60) -> None:
    """
    Force a hard commit directly to Solr. This is independent from any API endpoint.
    """
    url = f"{solr_base.rstrip('/')}/{core}/update"
    r = requests.post(url, params={"commit": "true", "wt": "json"}, json={}, timeout=timeout)
    if r.status_code >= 300:
        raise RuntimeError(f"Solr hard commit failed: {r.status_code} {r.text[:1000]}")


def solr_doc_count(solr_base: str, core: str, timeout: int = 60) -> int:
    url = f"{solr_base.rstrip('/')}/{core}/select"
    r = requests.get(url, params={"q": "*:*", "rows": 0, "wt": "json"}, timeout=timeout)
    if r.status_code >= 300:
        raise RuntimeError(f"Solr count failed: {r.status_code} {r.text[:1000]}")
    return int(r.json()["response"]["numFound"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to normalised_data.jsonl")
    ap.add_argument("--api", default="http://localhost:8000", help="API base URL")
    ap.add_argument("--solr", default="http://localhost:8983/solr", help="Solr base URL")
    ap.add_argument("--core", required=True, help="Solr core name")
    ap.add_argument("--batch", type=int, default=250, help="Docs per API batch")
    ap.add_argument("--commit-each", action="store_true", help="Commit each batch (slow, safest)")
    ap.add_argument("--commit-within-ms", type=int, default=10000, help="Solr commitWithin for batches")
    ap.add_argument("--final-solr-commit", action="store_true", help="Hard commit at end directly to Solr")
    ap.add_argument("--resume-from", type=int, default=0, help="Skip first N docs (resume)")
    ap.add_argument("--max-docs", type=int, default=0, help="Only ingest N docs (0 = all)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep between batches (seconds)")
    ap.add_argument("--no-count", action="store_true", help="Skip pre-counting for % progress")
    args = ap.parse_args()

    ingest_url = f"{args.api.rstrip('/')}/ingest_batch/{args.core}"

    total_lines: Optional[int] = None
    if not args.no_count:
        total_lines = count_lines(args.file)
        # if resuming, remaining is smaller; still useful for %
        if args.resume_from > 0:
            total_lines = max(total_lines - args.resume_from, 0)
        if args.max_docs and total_lines is not None:
            total_lines = min(total_lines, args.max_docs)

    seen = 0
    start_ts = time.time()

    def fmt_progress(done: int) -> str:
        if not total_lines:
            return f"{done} docs"
        pct = (done / total_lines * 100.0) if total_lines else 0.0
        elapsed = max(time.time() - start_ts, 0.001)
        rate = done / elapsed
        remaining = max(total_lines - done, 0)
        eta = remaining / rate if rate > 0 else 0.0
        return f"{done}/{total_lines} ({pct:.1f}%) | {rate:.1f} docs/s | ETA {eta/60:.1f} min"

    for batch_docs in chunks(iter_jsonl(args.file, args.resume_from, args.max_docs), args.batch):
        payload: Dict[str, Any] = {
            "docs": batch_docs,
            "commit": bool(args.commit_each),
            "commit_within_ms": args.commit_within_ms,
        }

        r = requests.post(ingest_url, json=payload, timeout=180)
        if r.status_code >= 300:
            print("ERROR:", r.status_code, r.text[:1000], file=sys.stderr)
            sys.exit(1)

        seen += len(batch_docs)
        print("Indexed:", fmt_progress(seen))

        if args.sleep > 0:
            time.sleep(args.sleep)

    # Optional hard commit to Solr (recommended for big ingest)
    if args.final_solr_commit and not args.commit_each:
        solr_hard_commit(args.solr, args.core, timeout=120)
        print("Final Solr hard commit: OK")

    # Quick verification
    try:
        n = solr_doc_count(args.solr, args.core, timeout=60)
        print(f"Solr doc count in core '{args.core}': {n}")
    except Exception as e:
        print(f"WARNING: could not verify Solr count: {e}", file=sys.stderr)

    print("Done. Total ingested:", seen)


if __name__ == "__main__":
    main()

