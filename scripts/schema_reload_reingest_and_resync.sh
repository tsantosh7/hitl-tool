#!/usr/bin/env bash
set -euo pipefail

CORE="${CORE:-hitl_test}"
SOLR_BASE="${SOLR_BASE:-http://localhost:8983/solr}"
API_URL="${API_URL:-http://localhost:8000}"

log() { echo "[$(date -Is)] $*"; }

schema_api() {
  curl -sS -H 'Content-Type: application/json' \
    -X POST "${SOLR_BASE}/${CORE}/schema" \
    -d "$1"
}

log "0) Sanity: Solr core ping"
curl -sS "${SOLR_BASE}/${CORE}/admin/ping?wt=json" >/dev/null
log "   OK"

log "1) Add fields via Schema API (idempotent; ignore 'already exists')"

add_field() {
  local name="$1"
  local type="$2"
  local multi="$3"
  local indexed="$4"
  local stored="$5"

  # Check if field exists
  if curl -sS "${SOLR_BASE}/${CORE}/schema/fields/${name}?wt=json" | grep -q "\"name\":\"${name}\""; then
    log "   Field exists: ${name}"
    return 0
  fi

  log "   Adding field: ${name}"
  schema_api "{
    \"add-field\": {
      \"name\": \"${name}\",
      \"type\": \"${type}\",
      \"multiValued\": ${multi},
      \"indexed\": ${indexed},
      \"stored\": ${stored}
    }
  }" >/dev/null
}

add_field "codes_v1_ss"  "string" "true" "true" "true"
add_field "codes_ext_ss" "string" "true" "true" "true"
add_field "codes_all_ss" "string" "true" "true" "true"

log "2) Reload core to ensure everything is applied cleanly"
curl -sS "${SOLR_BASE}/admin/cores?action=RELOAD&core=${CORE}&wt=json" >/dev/null
log "   Reloaded: ${CORE}"

log "3) Verify schema now contains fields"
for f in codes_v1_ss codes_ext_ss codes_all_ss; do
  curl -sS "${SOLR_BASE}/${CORE}/schema/fields/${f}?wt=json" | grep -q "\"name\":\"${f}\""
  log "   Present: ${f}"
done

log "4) Re-ingest corpus (you said you're OK with this)"
./run_ingestion.sh

log "5) Trigger hypothesis sync (stream)"
curl -N -sS -X POST "${API_URL}/hypothesis/sync_stream" \
  -H "Content-Type: application/json" \
  -d "{\"core\":\"${CORE}\",\"all_groups\":true,\"only_enabled_groups\":true,\"write_snapshot\":true,\"limit_per_request\":200,\"force_full\":true}" \
| sed -n '1,200p'

log "6) Quick query checks (won't 400 if schema is correct)"
curl -sS "${SOLR_BASE}/${CORE}/select?q=codes_all_ss:*&rows=0&wt=json" | sed -n '1,120p' || true

log "Done."
