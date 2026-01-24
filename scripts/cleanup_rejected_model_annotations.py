import os
import argparse
import time
from typing import Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HYPOTHESIS_API_BASE = "https://api.hypothes.is/api"


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        status=8,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "DELETE"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def hyp_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.hypothesis.v1+json",
        "Content-Type": "application/json;charset=utf-8",
    }


def hyp_get_profile(session: requests.Session, token: str) -> dict:
    r = session.get(f"{HYPOTHESIS_API_BASE}/profile", headers=hyp_headers(token), timeout=60)
    r.raise_for_status()
    return r.json()


def hyp_search_rows(
    session: requests.Session,
    token: str,
    *,
    group: str,
    limit: int = 200,
    sort: str = "updated",
    order: str = "desc",
    tags: Optional[List[str]] = None,
    references: Optional[str] = None,
) -> List[dict]:
    params: Dict[str, object] = {"group": group, "limit": int(limit), "sort": sort, "order": order}
    if tags:
        params["tag"] = tags
    if references:
        params["references"] = references

    r = session.get(f"{HYPOTHESIS_API_BASE}/search", params=params, headers=hyp_headers(token), timeout=60)
    r.raise_for_status()
    return r.json().get("rows", []) or []


def hyp_delete_annotation(session: requests.Session, token: str, ann_id: str) -> None:
    r = session.delete(f"{HYPOTHESIS_API_BASE}/annotations/{ann_id}", headers=hyp_headers(token), timeout=60)
    if r.status_code not in (200, 204, 404):
        r.raise_for_status()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.getenv("HYPOTHESIS_API_TOKEN", ""), help="Hypothesis API token")
    ap.add_argument("--group", required=True, help="Hypothesis group id")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--max-parents", type=int, default=5000)
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument(
        "--reject-tags",
        default="review:reject,review:incorrect,reject",
        help="Comma-separated tags that count as rejection (on human replies)",
    )
    ap.add_argument(
        "--base-tags",
        default="source:model,bot:hitl",
        help="Comma-separated tags that identify bot model annotations",
    )
    args = ap.parse_args()

    if not args.token.strip():
        raise SystemExit("ERROR: provide --token or set HYPOTHESIS_API_TOKEN")

    session = make_session()

    profile = hyp_get_profile(session, args.token)
    bot_userid = profile.get("userid") or profile.get("user") or profile.get("username")
    if not bot_userid:
        raise SystemExit("ERROR: could not determine bot userid from /profile")
    if isinstance(bot_userid, str) and "@hypothes.is" in bot_userid and not bot_userid.startswith("acct:"):
        bot_userid = f"acct:{bot_userid}"

    reject_tags = {t.strip() for t in args.reject_tags.split(",") if t.strip()}
    base_tags = [t.strip() for t in args.base_tags.split(",") if t.strip()]

    inspected = deleted = kept = skipped = 0

    parents = hyp_search_rows(
        session,
        args.token,
        group=args.group,
        tags=base_tags,
        limit=args.limit,
        sort="updated",
        order="desc",
    )

    for parent in parents:
        if inspected >= args.max_parents:
            break
        inspected += 1

        parent_id = parent.get("id")
        if not parent_id:
            skipped += 1
            continue

        if parent.get("user") != bot_userid:
            skipped += 1
            continue

        replies = hyp_search_rows(
            session,
            args.token,
            group=args.group,
            references=parent_id,
            limit=args.limit,
            sort="updated",
            order="desc",
        )

        rejected = False
        for r in replies:
            u = r.get("user")
            if not u or u == bot_userid:
                continue
            tset = set(r.get("tags") or [])
            if any(t in tset for t in reject_tags):
                rejected = True
                break

        if rejected:
            if args.dry_run:
                print(f"[dry-run] would delete bot annotation {parent_id} (rejected by human reply)")
            else:
                hyp_delete_annotation(session, args.token, parent_id)
            deleted += 1
        else:
            kept += 1

        if args.sleep:
            time.sleep(args.sleep)

    print("\n✅ REJECT CLEANUP DONE")
    print(f"bot_userid={bot_userid}")
    print(f"inspected={inspected} deleted={deleted} kept={kept} skipped={skipped}")


if __name__ == "__main__":
    main()
