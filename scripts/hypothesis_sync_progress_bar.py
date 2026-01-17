import json
import sys
import time
import requests

API = "http://localhost:8000"
URL = f"{API}/hypothesis/sync_stream"

payload = {
    "core": "hitl_test",
    "all_groups": True,
    "only_enabled_groups": True,
    "write_snapshot": True,
    "limit_per_request": 200,
    "force_full": False,
}

def bar(done: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[" + ("?" * width) + "]"
    done = max(0, min(done, total))
    filled = int(width * done / total)
    return "[" + ("=" * filled) + (" " * (width - filled)) + "]"

groups_total = 0
groups_done = 0
group_i = 0
group_id = ""
phase = ""
group_fetched = 0
seen = 0
linked = 0
flagged = 0

last_render = 0.0

with requests.post(URL, json=payload, stream=True, timeout=(10, 600)) as r:
    r.raise_for_status()

    for line in r.iter_lines(decode_unicode=True):
        if line is None:
            continue
        line = line.strip()
        if not line or line.startswith(":"):
            continue

        if line.startswith("event:"):
            continue

        if not line.startswith("data:"):
            continue

        data = line[len("data:"):].strip()
        try:
            obj = json.loads(data)
        except Exception:
            continue

        # progress payloads
        if "phase" in obj and "groups_total" in obj:
            phase = obj.get("phase", phase)
            groups_total = obj.get("groups_total", groups_total)
            groups_done = obj.get("groups_done", groups_done)
            group_i = obj.get("group_i", group_i)
            group_id = obj.get("group_id", group_id)
            group_fetched = obj.get("group_annotations_fetched", group_fetched)

            seen = obj.get("annotations_seen", seen)
            linked = obj.get("annotations_linked_to_docs", linked)
            flagged = obj.get("docs_flagged_in_solr", flagged)

        # final result
        if obj.get("ok") is True and "groups_synced" in obj:
            groups_done = obj.get("groups_synced", groups_done)
            seen = obj.get("annotations_seen", seen)
            linked = obj.get("annotations_linked_to_docs", linked)
            flagged = obj.get("docs_flagged_in_solr", flagged)
            phase = "done"

        # throttle rendering a bit
        now = time.time()
        if now - last_render < 0.1:
            continue
        last_render = now

        sys.stdout.write(
            "\r"
            f"{bar(groups_done, groups_total)} "
            f"{groups_done}/{groups_total} groups | "
            f"{phase} g{group_i}/{groups_total} {group_id} "
            f"fetched={group_fetched} | "
            f"seen={seen} linked={linked} flagged={flagged}    "
        )
        sys.stdout.flush()

        if phase == "done":
            break

print("\nDone.")
