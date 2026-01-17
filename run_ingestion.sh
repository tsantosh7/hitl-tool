#!/usr/bin/env bash
set -euo pipefail
source ~/environments/juddges_env/bin/activate

python scripts/ingest_jsonl.py \
  --file /home/stirunag/work/github/hitl-tool/data/normalised_data.jsonl \
  --api http://localhost:8000 \
  --solr http://localhost:8983/solr \
  --core hitl_test \
  --batch 250 \
  --commit-within-ms 10000 \
  --final-solr-commit
