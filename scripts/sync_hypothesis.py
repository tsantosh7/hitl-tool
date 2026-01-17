# scripts/sync_hypothesis.py
import os
import json
import argparse
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import requests


HYPOTHESIS_API_BASE = "https://api.hypothes.is/api"


def hyp_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.hypothesis.v1+json",
        "Content-Type": "application/json;charset=utf-8",
    }


def get_profile(token: str) -> dict:
    r = requests.get(f"{HYPOTHESIS_API_BASE}/profile", headers=hyp_headers(token), timeout=60)
    r.raise_for_status()
    return r.json()


def iter_group_annotations(token: str, group_id: str, limit: int = 200) -> Iterable[dict]:
    params = {"group": group_id, "sort": "updated", "order": "asc", "limit": int(limit)}
    search_after = None
    while True:
        if search_after:
            params["search_after"] = search_after
        r = requests.get(f"{HYPOTHESIS_API_BASE}/search", params=params, headers=hyp_headers(token), timeout=60)
        r.raise_for_status()
        data = r.json()
        rows = data.get("rows", [])
        if not rows:
            break
        for a in rows:
            yield a
        search_after = rows[-1].get("updated")
        if not search_after:
            break


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.getenv("HYPOTHESIS_API_TOKEN", ""), help="Hypothesis API token")
    ap.add_argument("--api", default="http://localhost:8000", help="Your FastAPI base URL")
    ap.add_argument("--core", default="hitl_test", help="Solr core")
    ap.add_argument("--outdir", default="data/hypothesis", help="Base output dir for snapshots")
    ap.add_argument("--limit", type=int, default=200, help="Hypothesis API page size")
    args = ap.parse_args()

    if not args.token.strip():
        raise SystemExit("ERROR: provide --token or set HYPOTHESIS_API_TOKEN")

    profile = get_profile(args.token)
    groups = profile.get("groups", []) or []
    group_ids = [g.get("id") for g in groups if g.get("id")]

    day = datetime.utcnow().strftime("%Y-%m-%d")
    day_dir = os.path.join(args.outdir, day)
    os.makedirs(day_dir, exist_ok=True)

    total = 0
    for gid in group_ids:
        path = os.path.join(day_dir, f"group_{gid}.jsonl")
        rows = list(iter_group_annotations(args.token, gid, limit=args.limit))
        n = write_jsonl(path, rows)
        total += n
        print(f"[snapshot] group={gid} annotations={n} -> {path}")

        # Import snapshot into API (DB upsert + Solr flags)
        r = requests.post(
            f"{args.api.rstrip('/')}/hypothesis/import_snapshot",
            json={"core": args.core, "snapshot_path": os.path.abspath(path)},
            timeout=300,
        )
        if r.status_code >= 300:
            print("IMPORT ERROR:", r.status_code, r.text[:1000])
            raise SystemExit(1)

        print(f"[import] group={gid} ok")

    print("Done. Total annotations snapshotted:", total)


if __name__ == "__main__":
    main()
