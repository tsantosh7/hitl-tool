# scripts/sync_hypothesis_stream_client.py
import argparse
import json
import sys
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--core", default="hitl_test")
    ap.add_argument("--all-groups", action="store_true", default=True)
    ap.add_argument("--write-snapshot", action="store_true", default=True)
    ap.add_argument("--force-full", action="store_true", default=False)
    args = ap.parse_args()

    payload = {
        "core": args.core,
        "all_groups": True,
        "write_snapshot": bool(args.write_snapshot),
        "only_enabled_groups": True,
        "force_full": bool(args.force_full),
    }

    url = f"{args.api.rstrip('/')}/hypothesis/sync_stream"
    with requests.post(url, json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()

        groups_total = None
        current_group = None
        fetched = 0

        for raw_line in r.iter_lines(decode_unicode=True):
            if not raw_line:
                continue

            if raw_line.startswith("event:"):
                event = raw_line.split(":", 1)[1].strip()
                continue

            if raw_line.startswith("data:"):
                data = raw_line.split(":", 1)[1].strip()
                d = json.loads(data)

                if event == "group_start":
                    current_group = d["group_id"]
                    groups_total = (d["n"], d["i"])
                    fetched = 0
                    print(f"\n==> Group {d['i']}/{d['n']} {current_group} (cursor={d.get('cursor')})")

                elif event == "group_progress":
                    fetched = d.get("annotations_fetched", fetched)
                    sys.stdout.write(f"\r   fetched: {fetched}")
                    sys.stdout.flush()

                elif event == "group_done":
                    print(f"\n   done: seen={d['annotations_seen']} linked={d['linked']} docs_flagged={d['docs_flagged']} new_cursor={d.get('new_cursor')}")

                elif event == "result":
                    print("\n\n✅ FINAL RESULT")
                    print(json.dumps(d, indent=2))
                    return

                elif event == "error":
                    print("\n\n❌ ERROR")
                    print(json.dumps(d, indent=2))
                    return


if __name__ == "__main__":
    main()
