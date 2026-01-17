#!/usr/bin/env bash
set -euo pipefail
#How to use

#Fresh Solr rebuild only (keep Postgres):
#./reindex_core_and_reingest.sh

#Full clean rebuild (wipe Postgres + rebuild Solr + ingest):
#WIPE_DB=true ./reindex_core_and_reingest.sh

#Change core name:
#CORE=hitl_prod ./reindex_core_and_reingest.sh

#
# ----------------------------
# Config (override via env)
# ----------------------------
CORE="${CORE:-hitl_test}"
API_URL="${API_URL:-http://localhost:8000}"
SOLR_BASE="${SOLR_BASE:-http://localhost:8983/solr}"
CONFIGSET="${CONFIGSET:-hitl_configset}"

# If true, will wipe Postgres documents/projects/annotations tables before ingest.
# WARNING: This deletes data. Use only when you want a fresh rebuild.
WIPE_DB="${WIPE_DB:-false}"

# ----------------------------
# Helpers
# ----------------------------
log() { echo "[$(date -Is)] $*"; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 1; }
}

curl_json() {
  curl -sS "$@" -H "Accept: application/json"
}

# ----------------------------
# Preflight
# ----------------------------
require_cmd curl
require_cmd docker

log "Using CORE=$CORE API_URL=$API_URL SOLR_BASE=$SOLR_BASE CONFIGSET=$CONFIGSET WIPE_DB=$WIPE_DB"

log "Checking API health..."
curl_json "$API_URL/health" >/dev/null || { echo "API health check failed" >&2; exit 1; }

log "Checking Solr is reachable..."
curl_json "$SOLR_BASE/admin/info/system" >/dev/null || { echo "Solr system check failed" >&2; exit 1; }

# ----------------------------
# Optionally wipe DB
# ----------------------------
if [[ "$WIPE_DB" == "true" ]]; then
  log "WIPE_DB=true: wiping Postgres tables (documents, projects, annotations)..."

  # Uses env vars inside the container (POSTGRES_USER/POSTGRES_DB)
  docker exec -i hitl-tool-postgres-1 sh -lc '
    set -e
    psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 <<SQL
      -- Order matters due to FKs
      TRUNCATE TABLE
        hypothesis_annotations,
        project_documents,
        projects,
        documents
      RESTART IDENTITY CASCADE;
SQL
  '
  log "Postgres wiped."
else
  log "WIPE_DB=false: leaving Postgres as-is."
fi

# ----------------------------
# Delete + recreate Solr core
# ----------------------------
log "Unloading Solr core '$CORE' (if it exists)..."
curl_json -X POST "$SOLR_BASE/admin/cores?action=UNLOAD&core=$CORE&deleteIndex=true&deleteDataDir=true&deleteInstanceDir=true" >/dev/null || true
log "Unload attempted."

log "Creating Solr core '$CORE' with configSet '$CONFIGSET'..."
# If your Solr is not in cloud mode, this should work for core creation.
# (Your stack already uses configsets, so it should be fine.)
curl_json -X POST "$SOLR_BASE/admin/cores?action=CREATE&name=$CORE&configSet=$CONFIGSET" >/dev/null

log "Core created. Waiting briefly for Solr to be ready..."
sleep 2

# ----------------------------
# Re-ingest
# ----------------------------
log "Running ingestion (./run_ingestion.sh)..."
./run_ingestion.sh

# ----------------------------
# Verify
# ----------------------------
log "Verifying Solr doc count..."
curl_json "$SOLR_BASE/$CORE/select?q=*:*&rows=0" | sed -n '1,120p'

log "Done."
