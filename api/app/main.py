# api/app/main.py
import os
import json
import random
import re
import requests
import urllib.parse
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Iterable, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, HttpUrl
from sqlalchemy import select

import queue
import threading

from datetime import timezone
import uuid

from sqlalchemy import func


from uuid import UUID
import csv
import io
from fastapi.responses import StreamingResponse
from sqlalchemy import func


import urllib.parse

from pydantic import BaseModel, Field



from .db import SessionLocal
from .init_db import init
from .models import (
    Team, Project, Document, ProjectDocument,
    HypothesisGroup, HypothesisAnnotation,
    Code, CodeAlias,
)


from typing import Optional, Tuple

SOLR_BASE_URL = os.getenv("SOLR_BASE_URL", "").strip()
HYPOTHESIS_API_TOKEN = os.getenv("HYPOTHESIS_API_TOKEN", "").strip()
DATA_DIR = os.getenv("DATA_DIR", "").strip() or os.path.join(os.getcwd(), "data")
HYPOTHESIS_SNAPSHOT_DIR = os.path.join(DATA_DIR, "hypothesis")
HYPOTHESIS_API_BASE = "https://api.hypothes.is/api"

# Global-core model: projects point to one shared Solr core
SOLR_GLOBAL_CORE = os.getenv("SOLR_GLOBAL_CORE", "hitl_test").strip() or "hitl_test"

# ------------------------------------------------------------------------------
# Hypothesis public group safety (NEVER sync __world__ unless explicitly requested)
# ------------------------------------------------------------------------------
HYPOTHESIS_PUBLIC_GROUP_ID = "__world__"
HYPOTHESIS_EXCLUDE_PUBLIC = os.getenv("HYPOTHESIS_EXCLUDE_PUBLIC", "true").lower() == "true"

app = FastAPI()


@app.on_event("startup")
def on_startup():
    init()
    os.makedirs(HYPOTHESIS_SNAPSHOT_DIR, exist_ok=True)
    db = SessionLocal()
    try:
        seed_v1_codes(db)
        seed_code_aliases(db)
    finally:
        db.close()



@app.get("/health")
def health():
    return {
        "ok": True,
        "solr": SOLR_BASE_URL,
        "solr_global_core": SOLR_GLOBAL_CORE,
        "data_dir": DATA_DIR,
        "hypothesis_snapshot_dir": HYPOTHESIS_SNAPSHOT_DIR,
        "has_hypothesis_token": bool(HYPOTHESIS_API_TOKEN),
    }


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def parse_dt_utc(val):
    """
    Normalize Hypothesis timestamps to timezone-aware UTC datetimes.
    Accepts ISO8601 strings (with Z or +00:00) or datetime objects.
    Returns None if val is falsy.
    """
    if not val:
        return None

    if isinstance(val, datetime):
        dt = val
    else:
        s = str(val).strip()
        # Hypothesis sometimes uses Z; fromisoformat wants +00:00
        s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _require_solr() -> None:
    if not SOLR_BASE_URL:
        raise HTTPException(status_code=500, detail="SOLR_BASE_URL is not set on the API server.")


def normalize_url(u: Optional[str]) -> Optional[str]:
    """
    Normalizes URLs so Hypothesis target URLs match ingested canonical URLs.
    - strip fragment
    - lowercase scheme + host
    - remove trailing slash except root
    """
    if not u:
        return None
    u = str(u).strip()
    if not u:
        return None
    try:
        p = urllib.parse.urlsplit(u)
        scheme = (p.scheme or "http").lower()
        netloc = (p.netloc or "").lower()
        path = p.path or "/"
        if path != "/" and path.endswith("/"):
            path = path[:-1]
        # drop fragments
        return urllib.parse.urlunsplit((scheme, netloc, path, p.query, ""))
    except Exception:
        return u.split("#", 1)[0]


def solr_core_url(core: str) -> str:
    _require_solr()
    return f"{SOLR_BASE_URL.rstrip('/')}/{core}"


def utc_now_z() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def date_to_solr_dt(d: Optional[date]) -> Optional[str]:
    if not d:
        return None
    return f"{d.isoformat()}T00:00:00Z"


def chunked(xs: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


# ------------------------------------------------------------------------------
# Annotations helpers - code normalisation
# ------------------------------------------------------------------------------

V1_CODES_PATH = os.getenv("V1_CODES_PATH", "/app/data/schema_v1_codes.json")


def load_v1_codes() -> set[str]:
    """
    Loads the canonical v1 list (original 43) from JSON.
    Expected format: {"v1_codes": ["CodeA", "CodeB", ...]}
    """
    try:
        with open(V1_CODES_PATH, "r", encoding="utf-8") as f:
            obj = json.load(f)
        codes = obj.get("v1_codes") or []
        return {str(c).strip() for c in codes if str(c).strip()}
    except Exception:
        return set()


V1_CODES: set[str] = load_v1_codes()



def seed_v1_codes(db):
    """
    Ensures all v1 codes exist in the DB and are locked.
    Safe to run on every startup.
    """
    if not V1_CODES:
        return

    existing = set(
        db.execute(select(Code.code).where(Code.version == "v1")).scalars().all()
    )
    missing = [c for c in V1_CODES if c not in existing]

    for c in missing:
        db.add(
            Code(
                code=c,
                version="v1",
                is_active=True,
                is_locked=True,
            )
        )

    if missing:
        db.commit()




def normalize_tag(t: str) -> str:
    """Light trim only (we keep raw tag for audit; normalization happens in key funcs)."""
    return (t or "").strip()


def canon_key(s: str) -> str:
    """
    A stable comparison key:
    - lowercased
    - remove all non-alphanumerics
    This makes tags like "Confess/Plead" and "confess plead" comparable.
    """
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def levenshtein(a: str, b: str, max_dist: int = 2) -> int:
    """
    Bounded Levenshtein distance.
    Returns > max_dist if distance exceeds threshold (fast exit).
    """
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        row_min = cur[0]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            v = min(ins, dele, sub)
            cur.append(v)
            if v < row_min:
                row_min = v
        prev = cur
        if row_min > max_dist:
            return max_dist + 1
    return prev[-1]


def load_code_maps(db) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """
    Returns:
      - canon_version: canonical_code -> "v1"|"ext" (only active codes)
      - alias_to_canon: alias -> canonical_code
      - key_to_canon: canon_key(canonical_code) -> canonical_code (active only)
    """
    code_rows = db.execute(select(Code.code, Code.version, Code.is_active)).all()
    canon_version: dict[str, str] = {}
    key_to_canon: dict[str, str] = {}

    for code, version, active in code_rows:
        if not active:
            continue
        canon_version[code] = version
        key_to_canon[canon_key(code)] = code

    alias_rows = db.execute(select(CodeAlias.alias, CodeAlias.code)).all()
    alias_to_canon = {normalize_tag(a): c for (a, c) in alias_rows}

    return canon_version, alias_to_canon, key_to_canon


def resolve_tag_to_canonical(
    tag: str,
    canon_version: dict[str, str],
    alias_to_canon: dict[str, str],
    key_to_canon: dict[str, str],
    *,
    fuzzy_max_dist: int = 2,
) -> Optional[str]:
    """
    Resolve a raw Hypothesis tag to a canonical registry code.
    Policy: if we can't confidently resolve, return None (treated as unregistered).
    """
    raw = normalize_tag(tag)
    if not raw:
        return None

    # 1) Exact canonical match
    if raw in canon_version:
        return raw

    # 2) Exact alias match
    if raw in alias_to_canon:
        return alias_to_canon[raw]

    # 3) Normalized key match (punctuation/case differences)
    k = canon_key(raw)
    if k in key_to_canon:
        return key_to_canon[k]

    # 4) Tiny fuzzy match against canonical keys (typos like Appellent)
    # Only accept if best is unique and within threshold
    best_code = None
    best_dist = fuzzy_max_dist + 1
    second_best = fuzzy_max_dist + 1

    for ck, canonical in key_to_canon.items():
        d = levenshtein(k, ck, max_dist=fuzzy_max_dist)
        if d < best_dist:
            second_best = best_dist
            best_dist = d
            best_code = canonical
        elif d < second_best:
            second_best = d

    if best_code is not None and best_dist <= fuzzy_max_dist and best_dist < second_best:
        return best_code

    return None


def split_codes(db, tags: list[str]) -> tuple[set[str], set[str], set[str]]:
    """
    Registry-backed split:
      - v1: canonical codes registered as version="v1"
      - ext: canonical codes registered as version="ext"
      - allc: union
    Unregistered/unresolvable tags are ignored (governance-first).
    """
    canon_version, alias_to_canon, key_to_canon = load_code_maps(db)

    v1: set[str] = set()
    ext: set[str] = set()
    allc: set[str] = set()

    for t in (tags or []):
        canonical = resolve_tag_to_canonical(
            t, canon_version, alias_to_canon, key_to_canon
        )
        if not canonical:
            continue

        allc.add(canonical)
        if canon_version.get(canonical) == "v1":
            v1.add(canonical)
        else:
            ext.add(canonical)

    return v1, ext, allc



def seed_code_aliases(db):
    """
    One-time / idempotent seeding of known legacy Hypothesis tags -> canonical codes.
    Safe to run multiple times.
    """
    mappings = {
        # safe + clear
        "Appellent": "Appellant",
        "Confess/Plead": "ConfessPleadGuilty",
        "WhatAncilliary": "WhatAncillary",
        "Ancillary": "WhatAncillary",
        "AquitOffence": "AcquitOffence",
        "ReasonSentExcess": "ReasonSentExcessNotLenient",
        "ReasonSentLenient": "ReasonSentLenientNotExcess",

        # OPTIONAL (uncomment if you're sure these are intended)
        # "ConvCourtType": "ConvCourtName",
        # "SentCourtType": "SentCourtName",
        # "RSE_is": "ReasonSentExcessNotLenient",
    }

    # only add aliases if target canonical exists
    existing_codes = set(db.execute(select(Code.code)).scalars().all())

    for alias, canonical in mappings.items():
        if canonical not in existing_codes:
            continue
        alias = normalize_tag(alias)
        if not alias:
            continue
        if db.get(CodeAlias, alias):
            continue
        db.add(CodeAlias(alias=alias, code=canonical))

    db.commit()


#######################################################
# CSV export helpers
#######################################################
def iso_z(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    dtu = parse_dt_utc(dt)
    return dtu.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve_codes_for_tags_cached(
    tags: list[str],
    canon_version: dict[str, str],
    alias_to_canon: dict[str, str],
    key_to_canon: dict[str, str],
) -> list[tuple[str, str]]:
    """
    Resolve tags -> list of (canonical_code, code_version).
    Uses your same normalization rules as sync.
    Only returns registered active codes.
    """
    out: list[tuple[str, str]] = []
    for t in (tags or []):
        canonical = resolve_tag_to_canonical(
            t,
            canon_version,
            alias_to_canon,
            key_to_canon,
        )
        if not canonical:
            continue
        ver = canon_version.get(canonical)
        if not ver:
            continue
        out.append((canonical, ver))
    return out


def iter_project_document_ids(
    db,
    project_id: str,
    document_id: Optional[str] = None,
    document_ids: Optional[str] = None,
) -> list[str]:
    """
    Returns the target document_ids (scoped to the project membership).
    project_id is UUID in DB, so we normalize it here.

    - If document_id provided: verifies it belongs to project
    - If document_ids provided: intersects with project docs
    - Else: all project docs
    """
    try:
        pid = UUID(str(project_id))
    except Exception:
        raise HTTPException(400, "project_id must be a valid UUID")

    proj_doc_ids = db.execute(
        select(ProjectDocument.document_id).where(ProjectDocument.project_id == pid)
    ).scalars().all()
    proj_set = set(proj_doc_ids)

    if document_id:
        if document_id not in proj_set:
            return []
        return [document_id]

    if document_ids:
        requested = {d.strip() for d in document_ids.split(",") if d.strip()}
        return sorted(list(proj_set & requested))

    return sorted(list(proj_set))


def csv_safe_col(s: str) -> str:
    """
    Convert a canonical code into a stable CSV column name.
    e.g. "ConfessPleadGuilty" -> "code__ConfessPleadGuilty"
         "Confess/Plead" (shouldn't happen after canonical) -> "code__Confess_Plead"
    """
    base = re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_")
    return f"code__{base}"


def build_wide_aggregates(
    db,
    doc_ids: list[str],
    canon_version: dict[str, str],
    alias_to_canon: dict[str, str],
    key_to_canon: dict[str, str],
    *,
    version: str = "all",     # v1|ext|all
    code_filter: str | None = None,
) -> tuple[dict[str, dict[str, dict]], set[str]]:
    """
    Returns:
      - per_doc: doc_id -> canonical_code -> metrics dict
      - codes_seen: set of canonical_code
    Metrics dict contains:
      { "count": int, "latest_value": str|None, "latest_updated": datetime|None }
    """
    per_doc: dict[str, dict[str, dict]] = {}
    codes_seen: set[str] = set()

    stmt = select(
        HypothesisAnnotation.document_id,
        HypothesisAnnotation.tags,
        HypothesisAnnotation.text,
        HypothesisAnnotation.updated,
    ).where(HypothesisAnnotation.document_id.in_(doc_ids))

    for doc_id, tags, text, updated in db.execute(stmt).yield_per(5000):
        if not doc_id:
            continue

        resolved = resolve_codes_for_tags_cached(tags or [], canon_version, alias_to_canon, key_to_canon)
        if not resolved:
            continue

        val = (text or "").strip()
        upd = parse_dt_utc(updated)

        for canonical_code, code_ver in resolved:
            if code_filter and canonical_code != code_filter:
                continue
            if version != "all" and code_ver != version:
                continue

            codes_seen.add(canonical_code)

            doc_bucket = per_doc.setdefault(doc_id, {})
            rec = doc_bucket.get(canonical_code)
            if not rec:
                rec = {"count": 0, "latest_value": None, "latest_updated": None}
                doc_bucket[canonical_code] = rec

            rec["count"] += 1

            if val:
                if upd:
                    if rec["latest_updated"] is None or upd > rec["latest_updated"]:
                        rec["latest_updated"] = upd
                        rec["latest_value"] = val
                else:
                    if rec["latest_value"] is None:
                        rec["latest_value"] = val

    return per_doc, codes_seen

# ------------------------------------------------------------------------------
# Solr helpers
# ------------------------------------------------------------------------------

def solr_add_docs(
    core: str,
    docs: List[dict],
    commit: bool = True,
    commit_within_ms: Optional[int] = None,
) -> None:
    params: Dict[str, str] = {}
    if commit:
        params["commit"] = "true"
    if commit_within_ms is not None:
        params["commitWithin"] = str(int(commit_within_ms))

    r = requests.post(
        f"{solr_core_url(core)}/update",
        params=params,
        json=docs,
        timeout=180,
    )
    if r.status_code >= 300:
        raise HTTPException(status_code=500, detail=f"Solr add failed: {r.status_code} {r.text[:800]}")


def solr_atomic_update(core: str, atomic_docs: List[dict], commit_within_ms: int = 5000) -> None:
    r = requests.post(
        f"{solr_core_url(core)}/update",
        params={"commitWithin": str(int(commit_within_ms))},
        json=atomic_docs,
        timeout=120,
    )
    if r.status_code >= 300:
        raise HTTPException(status_code=500, detail=f"Solr atomic update failed: {r.status_code} {r.text[:800]}")


def solr_update_codes_only(core: str, doc_codes: Dict[str, Dict[str, set[str]]]) -> int:
    """
    Atomic updates for codes_* only.
    Uses 'set' to be deterministic and de-duplicate.
    """
    if not doc_codes:
        return 0

    atomic_docs = []
    for document_id, codes in doc_codes.items():
        atomic_docs.append({
            "document_id_s": document_id,
            "codes_v1_ss": {"set": sorted(codes["v1"])},
            "codes_ext_ss": {"set": sorted(codes["ext"])},
            "codes_all_ss": {"set": sorted(codes["all"])},
        })

    updated = 0
    for batch in chunked(atomic_docs, 500):
        solr_atomic_update(core, batch, commit_within_ms=5000)
        updated += len(batch)

    return updated


def solr_add_project_membership(core: str, project_id: str, document_ids: list[str]) -> int:
    if not document_ids:
        return 0
    atomic_docs = []
    for did in document_ids:
        atomic_docs.append({
            "document_id_s": did,
            "project_ids_ss": {"add": project_id},
        })
    updated = 0
    for batch in chunked(atomic_docs, 500):
        solr_atomic_update(core, batch, commit_within_ms=5000)
        updated += len(batch)
    return updated


def solr_set_project_membership(core: str, doc_to_projects: dict[str, set[str]]) -> int:
    if not doc_to_projects:
        return 0

    atomic_docs = []
    for document_id, pids in doc_to_projects.items():
        atomic_docs.append({
            "document_id_s": document_id,
            "project_ids_ss": {"set": sorted(pids)},
        })

    updated = 0
    for batch in chunked(atomic_docs, 500):
        solr_atomic_update(core, batch, commit_within_ms=5000)
        updated += len(batch)
    return updated


def solr_select(core: str, params: dict) -> dict:
    """
    SOLR_BASE_URL may be:
      - http://solr:8983/solr        (common in docker)
      - http://solr:8983            (less common)
    This function handles both safely.
    """
    _require_solr()
    base = SOLR_BASE_URL.rstrip("/")

    # If base already ends with /solr, don't add another /solr
    if base.endswith("/solr"):
        url = f"{base}/{core}/select"
    else:
        url = f"{base}/solr/{core}/select"

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()



def solr_escape_term(s: str) -> str:
    # good enough for UUIDs + simple values
    return re.sub(r'([+\-!(){}[\]^"~*?:\\/]|&&|\|\|)', r'\\\1', s)


def normalize_fq_list(fq):
    if fq is None:
        return []
    if isinstance(fq, str):
        return [fq]
    return [x for x in fq if x]



# ------------------------------------------------------------------------------
# Corpus ingestion Solr doc builder (patched)
# ------------------------------------------------------------------------------

def to_solr_doc(payload: "IngestDocumentIn") -> dict:
    meta = payload.doc_metadata or {}
    if not isinstance(meta, dict):
        meta = {}

    def as_str(v) -> str:
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        return str(v)

    def as_str_list(v) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            out: List[str] = []
            for x in v:
                s = as_str(x).strip()
                if s:
                    out.append(s)
            return out
        s = as_str(v).strip()
        return [s] if s else []

    published_dt = date_to_solr_dt(payload.published_date)
    canonical = normalize_url(str(payload.canonical_url)) or str(payload.canonical_url)

    solr_doc = {
        "document_id_s": payload.document_id,
        "canonical_url_s": canonical,

        "title_txt": payload.title or "",
        "excerpt_txt": payload.excerpt or "",
        "body_txt": payload.content_text or "",

        "doc_type_s": payload.doc_type or "",
        "source_s": payload.source or "",

        "judges_ss": as_str_list(meta.get("judges")),
        "case_numbers_ss": as_str_list(meta.get("caseNumbers")),
        "citation_references_ss": as_str_list(meta.get("citation_references")),
        "legislation_ss": as_str_list(meta.get("legislation")),

        "citation_s": as_str(meta.get("citation")).strip(),
        "signature_s": as_str(meta.get("signature")).strip(),
        "xml_uri_s": as_str(meta.get("xml_uri")).strip(),
        "file_name_s": as_str(meta.get("file_name")).strip(),
        "appeal_type_s": as_str(meta.get("appeal_type")).strip(),
        "appeal_outcome_s": as_str(meta.get("appeal_outcome")).strip(),

        "schema_versions_ss": [payload.schema_version],

        # Always present so atomic updates + faceting behave consistently
        "project_ids_ss": [],

        "ingested_dt": utc_now_z(),
        "has_human_b": bool(payload.has_human),
        "has_model_b": bool(payload.has_model),
        "has_any_span_b": bool(payload.has_any_span),
        "rand_f": float(payload.rand_f) if payload.rand_f is not None else random.random(),
    }

    keep_empty = {"project_ids_ss", "schema_versions_ss"}
    for k in list(solr_doc.keys()):
        if k in keep_empty:
            continue
        if solr_doc[k] in ("", [], None):
            solr_doc.pop(k)

    if published_dt:
        solr_doc["published_dt"] = published_dt

    return solr_doc


# ------------------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------------------

class IngestDocumentIn(BaseModel):
    document_id: str
    canonical_url: HttpUrl

    published_date: Optional[date] = None
    doc_type: Optional[str] = None
    title: Optional[str] = None
    excerpt: Optional[str] = None
    content_text: str
    source: Optional[str] = None

    doc_metadata: Dict[str, Any] = Field(default_factory=dict, alias="metadata")

    schema_version: str = "hitl-v1"
    has_human: bool = False
    has_model: bool = False
    has_any_span: bool = False
    rand_f: Optional[float] = None

    model_config = {"populate_by_name": True, "extra": "ignore"}


class IngestBatchIn(BaseModel):
    docs: List[Dict[str, Any]]
    commit: bool = False
    commit_within_ms: int = 10_000


class HypothesisSyncRequest(BaseModel):
    core: str = SOLR_GLOBAL_CORE
    group_id: Optional[str] = None
    all_groups: bool = True
    only_enabled_groups: bool = True
    write_snapshot: bool = True
    limit_per_request: int = 200
    force_full: bool = False

    # NEW: make public syncing opt-in
    include_public: bool = False


# Optional: if your frontend expects these (Fix 2)
class CreateProjectIn(BaseModel):
    team_id: Optional[str] = None
    team_name: Optional[str] = None
    name: str = "Untitled Project"


class CreateProjectOut(BaseModel):
    team_id: str
    project_id: str
    solr_core: str


class TeamCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)

class ProjectCreateRequest(BaseModel):
    team_id: UUID
    name: str = Field(..., min_length=1)

class ProjectAddDocsRequest(BaseModel):
    document_ids: list[str] = Field(..., min_length=1)


# ------------------------------------------------------------------------------
# Ingestion endpoints (Fix 1)
# ------------------------------------------------------------------------------

@app.post("/ingest_batch/{core}")
def ingest_batch(core: str, payload: IngestBatchIn):
    """
    Matches scripts/ingest_jsonl.py which posts to /ingest_batch/{core}.
    Writes docs to Postgres (canonical) and Solr (search).
    """
    db = SessionLocal()
    try:
        to_index: List[dict] = []
        for d in payload.docs:
            doc_in = IngestDocumentIn(**d)

            canon = normalize_url(str(doc_in.canonical_url)) or str(doc_in.canonical_url)

            # --- canonical DB upsert ---
            row = db.get(Document, doc_in.document_id)
            if row:
                row.canonical_url = canon
                row.published_date = doc_in.published_date
                row.doc_type = doc_in.doc_type
                row.title = doc_in.title
                row.excerpt = doc_in.excerpt
                row.content_text = doc_in.content_text
                row.source = doc_in.source
                row.doc_metadata = doc_in.doc_metadata or {}
            else:
                row = Document(
                    document_id=doc_in.document_id,
                    canonical_url=canon,
                    published_date=doc_in.published_date,
                    doc_type=doc_in.doc_type,
                    title=doc_in.title,
                    excerpt=doc_in.excerpt,
                    content_text=doc_in.content_text,
                    source=doc_in.source,
                    doc_metadata=doc_in.doc_metadata or {},
                )
                db.add(row)

            to_index.append(to_solr_doc(doc_in))

        db.commit()

        solr_add_docs(
            core=core,
            docs=to_index,
            commit=bool(payload.commit),
            commit_within_ms=int(payload.commit_within_ms),
        )
        return {"ok": True, "core": core, "indexed": len(to_index), "commit": bool(payload.commit)}
    finally:
        db.close()


@app.post("/solr/{core}/commit")
def solr_commit(core: str):
    """
    Matches scripts calling POST /solr/{core}/commit.
    """
    r = requests.post(
        f"{solr_core_url(core)}/update",
        params={"commit": "true"},
        json=[],
        timeout=60,
    )
    if r.status_code >= 300:
        raise HTTPException(status_code=500, detail=f"Solr commit failed: {r.status_code} {r.text[:800]}")
    return {"ok": True, "core": core}


# ------------------------------------------------------------------------------
# Projects endpoints (Fix 2: global core model)
# ------------------------------------------------------------------------------

@app.post("/projects", response_model=CreateProjectOut)
def create_project(payload: CreateProjectIn):
    """
    Global core model: projects do NOT create per-project Solr cores.
    """
    db = SessionLocal()
    try:
        # Resolve / create team
        team = None
        if payload.team_id:
            team = db.get(Team, payload.team_id)
            if not team:
                raise HTTPException(status_code=404, detail=f"Team not found: {payload.team_id}")

        if not team:
            # Try find by name, else create
            team_name = (payload.team_name or "Default Team").strip()
            team = db.execute(select(Team).where(Team.name == team_name)).scalars().first()
            if not team:
                team = Team(team_id=str(uuid4()), name=team_name)
                db.add(team)
                db.commit()

        proj = Project(project_id=str(uuid4()), team_id=str(team.team_id), name=payload.name)
        db.add(proj)
        db.commit()

        # Global core: no core creation
        core_name = SOLR_GLOBAL_CORE

        return CreateProjectOut(
            team_id=str(team.team_id),
            project_id=str(proj.project_id),
            solr_core=core_name,
        )
    finally:
        db.close()


@app.delete("/projects/{project_id}")
def delete_project(project_id: str, delete_solr_core: bool = True):
    """
    Global core should never be deleted as part of a project delete.
    """
    db = SessionLocal()
    try:
        proj = db.get(Project, project_id)
        if not proj:
            raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

        core_name = SOLR_GLOBAL_CORE

        db.delete(proj)
        db.commit()

        return {"ok": True, "project_id": project_id, "solr_core": core_name, "deleted_solr_core": False}
    finally:
        db.close()


# ------------------------------------------------------------------------------
# Hypothesis helpers (patched: URL normalization + bulk resolve)
# ------------------------------------------------------------------------------

def _hyp_headers() -> Dict[str, str]:
    if not HYPOTHESIS_API_TOKEN:
        raise HTTPException(status_code=400, detail="HYPOTHESIS_API_TOKEN is not set on the API server.")
    return {
        "Authorization": f"Bearer {HYPOTHESIS_API_TOKEN}",
        "Accept": "application/vnd.hypothesis.v1+json",
        "Content-Type": "application/json;charset=utf-8",
    }


def hypothesis_get_profile() -> dict:
    r = requests.get(f"{HYPOTHESIS_API_BASE}/profile", headers=_hyp_headers(), timeout=60)
    if r.status_code >= 300:
        raise HTTPException(status_code=500, detail=f"Hypothesis profile failed: {r.status_code} {r.text[:800]}")
    return r.json()


def hypothesis_iter_group_annotations(
    group_id: str,
    limit: int = 200,
    search_after: Optional[str] = None,
) -> Iterable[dict]:
    """
    Incremental fetch using search_after on updated.
    """
    params = {
        "group": group_id,
        "sort": "updated",
        "order": "asc",
        "limit": int(limit),
    }
    if search_after:
        params["search_after"] = search_after

    while True:
        r = requests.get(f"{HYPOTHESIS_API_BASE}/search", params=params, headers=_hyp_headers(), timeout=60)
        if r.status_code >= 300:
            raise HTTPException(status_code=500, detail=f"Hypothesis search failed: {r.status_code} {r.text[:800]}")

        data = r.json()
        rows = data.get("rows", []) or []
        if not rows:
            break

        for a in rows:
            yield a

        last_updated = rows[-1].get("updated")
        if not last_updated:
            break
        params["search_after"] = last_updated


def snapshot_path_for_group(group_id: str) -> str:
    day = datetime.utcnow().strftime("%Y-%m-%d")
    day_dir = os.path.join(HYPOTHESIS_SNAPSHOT_DIR, day)
    os.makedirs(day_dir, exist_ok=True)
    return os.path.join(day_dir, f"group_{group_id}.jsonl")


def write_snapshot_jsonl(path: str, annotations: List[dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for a in annotations:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
            n += 1
    return n


def parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def hypothesis_extract(a: dict) -> Tuple[dict, bool, Optional[str]]:
    """
    Returns (fields, has_span, updated_string)
    """
    annotation_id = a.get("id")
    group_id = a.get("group")
    user = a.get("user")
    created_dt = parse_dt(a.get("created"))
    updated_str = a.get("updated")
    updated_dt = parse_dt(updated_str)

    text = a.get("text") or ""
    tags = a.get("tags") or []

    canonical_url = None
    exact = None
    prefix = None
    suffix = None

    targets = a.get("target") or []
    if targets:
        t0 = targets[0]
        canonical_url = normalize_url(t0.get("source"))
        selectors = t0.get("selector") or []
        for sel in selectors:
            if sel.get("type") == "TextQuoteSelector":
                exact = sel.get("exact")
                prefix = sel.get("prefix")
                suffix = sel.get("suffix")
                break

    has_span = bool(exact and str(exact).strip())

    fields = {
        "annotation_id": annotation_id,
        "group_id": group_id,
        "canonical_url": canonical_url,
        "user": user,
        "created": created_dt,
        "updated": updated_dt,
        "text": text,
        "tags": tags,
        "exact": exact,
        "prefix": prefix,
        "suffix": suffix,
        "raw": a,
    }
    return fields, has_span, updated_str


def upsert_group(db, g: dict) -> HypothesisGroup:
    """
    Safeguard: public groups (including __world__) default to disabled.
    Private groups default to enabled.
    Never auto-enable a group that was previously disabled.
    """
    gid = g.get("id")
    name = g.get("name") or gid or ""
    org = g.get("organization")
    scopes = g.get("scopes") or []

    is_public = (gid == HYPOTHESIS_PUBLIC_GROUP_ID) or bool(g.get("public"))
    default_enabled = (not is_public)

    row = db.get(HypothesisGroup, gid)
    if not row:
        row = HypothesisGroup(
            group_id=gid,
            name=name,
            organization=org,
            scopes=scopes,
            is_enabled=default_enabled,
        )
        # Hard-disable __world__ on insert if exclude is on
        if HYPOTHESIS_EXCLUDE_PUBLIC and gid == HYPOTHESIS_PUBLIC_GROUP_ID:
            row.is_enabled = False
        db.add(row)
        return row

    # Update metadata
    row.name = name or row.name
    row.organization = org
    row.scopes = scopes

    # Don't auto-enable previously disabled groups.
    # If nullable and currently unset, set default.
    if row.is_enabled is None:
        row.is_enabled = default_enabled

    # Hard-disable __world__ if exclude is on
    if HYPOTHESIS_EXCLUDE_PUBLIC and gid == HYPOTHESIS_PUBLIC_GROUP_ID:
        row.is_enabled = False

    return row


def bulk_resolve_document_ids(db, urls: List[str]) -> Dict[str, str]:
    """
    Resolve canonical_url -> document_id in bulk (with normalization).
    """
    url_to_doc: Dict[str, str] = {}
    if not urls:
        return url_to_doc

    normalized = [normalize_url(u) for u in urls]
    normalized = [u for u in normalized if u]
    if not normalized:
        return url_to_doc

    for batch in chunked(normalized, 1000):
        rows = db.execute(
            select(Document.canonical_url, Document.document_id).where(Document.canonical_url.in_(batch))
        ).all()
        for canon, doc_id in rows:
            url_to_doc[str(canon)] = str(doc_id)

    return url_to_doc


def upsert_annotations_bulk(
    db,
    fields_list: List[dict],
    url_to_doc: Dict[str, str],
) -> Tuple[int, int, Dict[str, Dict[str, bool]], Dict[str, Dict[str, set[str]]]]:
    """
    Upsert annotations. Returns:
      (annotations_seen, annotations_linked_to_docs, doc_flags_for_solr, doc_codes_for_solr)
    """
    seen = 0
    linked = 0
    doc_flags: Dict[str, Dict[str, bool]] = {}

    # document_id -> {"v1": set(), "ext": set(), "all": set()}
    doc_codes: Dict[str, Dict[str, set[str]]] = {}

    for fields in fields_list:
        seen += 1
        ann_id = fields["annotation_id"]

        canon = normalize_url(fields.get("canonical_url")) if fields.get("canonical_url") else None
        doc_id = url_to_doc.get(canon) if canon else None

        # Normalize timestamps *before* comparing/storing
        new_updated = parse_dt_utc(fields.get("updated"))
        new_created = parse_dt_utc(fields.get("created"))

        row = db.get(HypothesisAnnotation, ann_id)
        if row:
            row_updated = parse_dt_utc(row.updated)

            # If existing row is newer/equal, skip overwrite
            if row_updated and new_updated and new_updated <= row_updated:
                pass
            else:
                row.group_id = fields["group_id"]
                row.document_id = doc_id
                row.canonical_url = canon
                row.user = fields.get("user")
                row.created = new_created
                row.updated = new_updated
                row.text = fields.get("text")
                row.tags = fields.get("tags") or []
                row.exact = fields.get("exact")
                row.prefix = fields.get("prefix")
                row.suffix = fields.get("suffix")
                row.raw = fields.get("raw") or {}
        else:
            row = HypothesisAnnotation(
                annotation_id=ann_id,
                group_id=fields["group_id"],
                document_id=doc_id,
                canonical_url=canon,
                user=fields.get("user"),
                created=new_created,
                updated=new_updated,
                text=fields.get("text"),
                tags=fields.get("tags") or [],
                exact=fields.get("exact"),
                prefix=fields.get("prefix"),
                suffix=fields.get("suffix"),
                raw=fields.get("raw") or {},
            )
            db.add(row)

        if doc_id:
            linked += 1

            # Flags
            if doc_id not in doc_flags:
                doc_flags[doc_id] = {"has_human": True, "has_any_span": False}
            if fields.get("exact") and str(fields.get("exact")).strip():
                doc_flags[doc_id]["has_any_span"] = True

            # Codes (registry-backed)
            tags = fields.get("tags") or []
            v1, ext, allc = split_codes(db, tags)

            bucket = doc_codes.setdefault(doc_id, {"v1": set(), "ext": set(), "all": set()})
            bucket["v1"].update(v1)
            bucket["ext"].update(ext)
            bucket["all"].update(allc)

    return seen, linked, doc_flags, doc_codes




# def upsert_annotations_bulk(
#     db,
#     fields_list: List[dict],
#     url_to_doc: Dict[str, str],
# ) -> Tuple[int, int, Dict[str, Dict[str, bool]], Dict[str, Dict[str, set[str]]]]:
#     """
#     Upsert annotations. Returns:
#       (annotations_seen, annotations_linked_to_docs, doc_flags_for_solr, doc_codes_for_solr)
#     """
#     seen = 0
#     linked = 0
#     doc_flags: Dict[str, Dict[str, bool]] = {}
#
#     # document_id -> {"v1": set(), "ext": set(), "all": set()}
#     doc_codes: Dict[str, Dict[str, set[str]]] = {}
#
#     for fields in fields_list:
#         seen += 1
#         ann_id = fields["annotation_id"]
#
#         canon = normalize_url(fields.get("canonical_url")) if fields.get("canonical_url") else None
#         doc_id = url_to_doc.get(canon) if canon else None
#
#         row = db.get(HypothesisAnnotation, ann_id)
#         if row:
#             new_updated = fields.get("updated")
#             if row.updated and new_updated and new_updated <= row.updated:
#                 pass
#             else:
#                 row.group_id = fields["group_id"]
#                 row.document_id = doc_id
#                 row.canonical_url = canon
#                 row.user = fields.get("user")
#                 row.created = fields.get("created")
#                 row.updated = fields.get("updated")
#                 row.text = fields.get("text")
#                 row.tags = fields.get("tags") or []
#                 row.exact = fields.get("exact")
#                 row.prefix = fields.get("prefix")
#                 row.suffix = fields.get("suffix")
#                 row.raw = fields.get("raw") or {}
#         else:
#             row = HypothesisAnnotation(
#                 annotation_id=ann_id,
#                 group_id=fields["group_id"],
#                 document_id=doc_id,
#                 canonical_url=canon,
#                 user=fields.get("user"),
#                 created=fields.get("created"),
#                 updated=fields.get("updated"),
#                 text=fields.get("text"),
#                 tags=fields.get("tags") or [],
#                 exact=fields.get("exact"),
#                 prefix=fields.get("prefix"),
#                 suffix=fields.get("suffix"),
#                 raw=fields.get("raw") or {},
#             )
#             db.add(row)
#
#         if doc_id:
#             linked += 1
#
#             # Flags
#             if doc_id not in doc_flags:
#                 doc_flags[doc_id] = {"has_human": True, "has_any_span": False}
#             if fields.get("exact") and str(fields.get("exact")).strip():
#                 doc_flags[doc_id]["has_any_span"] = True
#
#             # Codes
#             tags = fields.get("tags") or []
#             v1, ext, allc = split_codes(tags)
#             bucket = doc_codes.setdefault(doc_id, {"v1": set(), "ext": set(), "all": set()})
#             bucket["v1"].update(v1)
#             bucket["ext"].update(ext)
#             bucket["all"].update(allc)
#
#     return seen, linked, doc_flags, doc_codes

def solr_update_flags_for_docs(
    core: str,
    doc_flags: Dict[str, Dict[str, bool]],
    doc_codes: Dict[str, Dict[str, set[str]]] | None = None,
) -> int:
    """
    Chunked Solr atomic updates.
    Returns number of docs updated.
    """
    doc_codes = doc_codes or {}

    if not doc_flags and not doc_codes:
        return 0

    doc_ids = set(doc_flags.keys()) | set(doc_codes.keys())

    atomic_docs = []
    for document_id in doc_ids:
        atomic = {"document_id_s": document_id}

        flags = doc_flags.get(document_id) or {}
        atomic["has_human_b"] = {"set": True}
        if "has_any_span" in flags:
            atomic["has_any_span_b"] = {"set": bool(flags["has_any_span"])}

        codes = doc_codes.get(document_id)
        if codes:
            atomic["codes_v1_ss"] = {"set": sorted(codes["v1"])}
            atomic["codes_ext_ss"] = {"set": sorted(codes["ext"])}
            atomic["codes_all_ss"] = {"set": sorted(codes["all"])}

        atomic_docs.append(atomic)

    updated = 0
    for batch in chunked(atomic_docs, 500):
        solr_atomic_update(core, batch, commit_within_ms=5000)
        updated += len(batch)

    return updated



# ------------------------------------------------------------------------------
# Progress streaming (SSE)
# ------------------------------------------------------------------------------

def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def run_hypothesis_sync(payload: HypothesisSyncRequest, emit=None) -> dict:
    """
    Shared implementation used by both /hypothesis/sync and /hypothesis/sync_stream.
    emit: optional function(event_name, data_dict) to push progress.
    """
    def _emit(ev: str, d: dict):
        if emit:
            emit(ev, d)

    db = SessionLocal()
    try:
        _emit("stage", {"msg": "fetch_profile"})
        profile = hypothesis_get_profile()
        groups = profile.get("groups", []) or []

        # Filter out Public group unless explicitly allowed (opt-in)
        exclude_public = HYPOTHESIS_EXCLUDE_PUBLIC and (not payload.include_public)
        if exclude_public:
            groups = [g for g in groups if g.get("id") != HYPOTHESIS_PUBLIC_GROUP_ID]

        _emit("stage", {"msg": "upsert_groups", "count": len(groups)})
        # for g in groups:
            # upsert_group(db, g)
        # db.commit()
        for g in groups:
            row = upsert_group(db, g)
            if payload.force_full and row:
                row.last_synced_updated = None
                row.last_synced_at = None
        db.commit()

        # Decide which groups to sync
        exclude_public = HYPOTHESIS_EXCLUDE_PUBLIC and (not payload.include_public)

        if payload.group_id and not payload.all_groups:
            if exclude_public and payload.group_id == HYPOTHESIS_PUBLIC_GROUP_ID:
                raise HTTPException(
                    status_code=400,
                    detail="Refusing to sync __world__ unless include_public=true."
                )
            group_ids = [payload.group_id]
        else:
            if payload.only_enabled_groups:
                enabled = (
                    db.execute(select(HypothesisGroup).where(HypothesisGroup.is_enabled == True))
                    .scalars()
                    .all()
                )
                group_ids = [g.group_id for g in enabled]
            else:
                group_ids = [g.get("id") for g in groups if g.get("id")]

        # Final hard filter (belt + suspenders)
        exclude_public = HYPOTHESIS_EXCLUDE_PUBLIC and (not payload.include_public)
        if exclude_public:
            group_ids = [gid for gid in group_ids if gid != HYPOTHESIS_PUBLIC_GROUP_ID]

        groups_total = len(group_ids)

        totals = {
            "groups_synced": 0,
            "annotations_seen": 0,
            "annotations_linked_to_docs": 0,
            "docs_flagged_in_solr": 0,
        }

        # Initial progress event (good for initializing a progress bar)
        _emit("progress", {
            "phase": "start",
            "groups_total": groups_total,
            "groups_done": 0,
            "annotations_seen": 0,
            "annotations_linked_to_docs": 0,
            "docs_flagged_in_solr": 0,
        })

        for gi, gid in enumerate(group_ids, start=1):
            g_row = db.get(HypothesisGroup, gid)
            cursor = None
            if g_row and not payload.force_full:
                cursor = g_row.last_synced_updated

            _emit("group_start", {"group_id": gid, "i": gi, "n": groups_total, "cursor": cursor})

            # Progress: group start
            _emit("progress", {
                "phase": "group_start",
                "group_id": gid,
                "group_i": gi,
                "groups_total": groups_total,
                "groups_done": totals["groups_synced"],
                "annotations_seen": totals["annotations_seen"],
                "annotations_linked_to_docs": totals["annotations_linked_to_docs"],
                "docs_flagged_in_solr": totals["docs_flagged_in_solr"],
            })

            ann_list: List[dict] = []
            last_updated_seen: Optional[str] = None

            # Fetch annotations (paginated)
            for raw in hypothesis_iter_group_annotations(
                gid,
                limit=payload.limit_per_request,
                search_after=cursor,
            ):
                ann_list.append(raw)
                last_updated_seen = raw.get("updated") or last_updated_seen

                # Emit periodic progress so clients can show activity
                if len(ann_list) % 500 == 0:
                    _emit("progress", {
                        "phase": "fetching",
                        "group_id": gid,
                        "group_i": gi,
                        "groups_total": groups_total,
                        "groups_done": totals["groups_synced"],
                        "group_annotations_fetched": len(ann_list),
                        "annotations_seen": totals["annotations_seen"],
                        "annotations_linked_to_docs": totals["annotations_linked_to_docs"],
                        "docs_flagged_in_solr": totals["docs_flagged_in_solr"],
                    })

            _emit("group_fetched", {"group_id": gid, "annotations_fetched": len(ann_list)})

            if payload.write_snapshot:
                path = snapshot_path_for_group(gid)
                write_snapshot_jsonl(path, ann_list)
                _emit("snapshot", {"group_id": gid, "path": path, "count": len(ann_list)})

            extracted: List[dict] = []
            urls: List[str] = []
            max_updated_str: Optional[str] = cursor

            for raw in ann_list:
                fields, _has_span, updated_str = hypothesis_extract(raw)
                extracted.append(fields)
                if fields.get("canonical_url"):
                    urls.append(fields["canonical_url"])
                if updated_str:
                    max_updated_str = updated_str  # sorted asc, so last wins

            urls_unique = list({normalize_url(u) for u in urls if u})
            urls_unique = [u for u in urls_unique if u]
            _emit("resolve_urls", {"group_id": gid, "unique_urls": len(urls_unique)})

            url_to_doc = bulk_resolve_document_ids(db, urls_unique)
            _emit("resolved", {"group_id": gid, "matched_docs": len(set(url_to_doc.values()))})

            seen, linked, doc_flags, doc_codes = upsert_annotations_bulk(db, extracted, url_to_doc)
            db.commit()
            updated_docs = solr_update_flags_for_docs(payload.core, doc_flags, doc_codes=doc_codes)

            _emit("codes_summary", {
                "group_id": gid,
                "docs_with_codes": len(doc_codes),
                "docs_with_flags": len(doc_flags),
                "sample_doc_id": next(iter(doc_codes.keys()), None),
                "sample_codes_all_count": (len(next(iter(doc_codes.values()))["all"]) if doc_codes else 0),
            })

            # if g_row and not payload.force_full and max_updated_str:
            #     g_row.last_synced_updated = max_updated_str
            #     g_row.last_synced_at = datetime.utcnow()
            #     db.commit()

            if g_row and max_updated_str:
                g_row.last_synced_updated = max_updated_str
                g_row.last_synced_at = datetime.utcnow()
                db.commit()

            totals["groups_synced"] += 1
            totals["annotations_seen"] += seen
            totals["annotations_linked_to_docs"] += linked
            totals["docs_flagged_in_solr"] += updated_docs

            # Progress: group done (this is what moves the bar)
            _emit("progress", {
                "phase": "group_done",
                "group_id": gid,
                "group_i": gi,
                "groups_total": groups_total,
                "groups_done": totals["groups_synced"],
                "annotations_seen": totals["annotations_seen"],
                "annotations_linked_to_docs": totals["annotations_linked_to_docs"],
                "docs_flagged_in_solr": totals["docs_flagged_in_solr"],
            })

            _emit("group_done", {
                "group_id": gid,
                "annotations_seen": seen,
                "linked": linked,
                "docs_flagged": updated_docs,
                "new_cursor": (g_row.last_synced_updated if g_row else None),
            })

        _emit("done", totals)
        return {
            "ok": True,
            "core": payload.core,
            **totals,
            "snapshot_dir": HYPOTHESIS_SNAPSHOT_DIR if payload.write_snapshot else None,
        }
    finally:
        db.close()


@app.post("/hypothesis/sync")
def hypothesis_sync(payload: HypothesisSyncRequest):
    return run_hypothesis_sync(payload)


@app.post("/hypothesis/sync_stream")
def hypothesis_sync_stream(payload: HypothesisSyncRequest):
    """
    Streams progress events (SSE) in real-time.
    """
    def gen():
        q: "queue.Queue[str]" = queue.Queue()
        done = threading.Event()

        def emit(ev: str, d: dict):
            q.put(sse_event(ev, d))

        def worker():
            try:
                result = run_hypothesis_sync(payload, emit=emit)
                q.put(sse_event("result", result))
            except Exception as e:
                q.put(sse_event("error", {"error": str(e)}))
            finally:
                done.set()

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        # Stream events as they arrive. Send a heartbeat so proxies don't buffer.
        while not done.is_set() or not q.empty():
            try:
                msg = q.get(timeout=1.0)
                yield msg
            except queue.Empty:
                # heartbeat comment line for SSE
                yield ": ping\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


############## Progress sync without hypothesis download##################
# Recompute codes for everything
# curl -sS -X POST "http://localhost:8000/solr/recompute_codes?core=hitl_test"
#
# Recompute codes only for one group
# curl -sS -X POST "http://localhost:8000/solr/recompute_codes?core=hitl_test&group_id=Qb9zgyQY"

@app.post("/solr/recompute_codes")
def recompute_solr_codes(core: str = "hitl_test", project_id: Optional[UUID] = None, group_id: Optional[str] = None):
    """
    Recompute Solr codes_* purely from Postgres hypothesis_annotations.
    No Hypothesis API calls.

    Optional filters:
      - project_id: only docs in that project
      - group_id: only annotations from that Hypothesis group
    """
    db = SessionLocal()
    try:
        # load code maps once
        canon_version, alias_to_canon, key_to_canon = load_code_maps(db)

        # determine doc scope (optional project filter)
        doc_scope: Optional[set[str]] = None
        if project_id:
            doc_scope = set(
                db.execute(select(ProjectDocument.document_id).where(ProjectDocument.project_id == project_id)).scalars().all()
            )

        stmt = select(
            HypothesisAnnotation.document_id,
            HypothesisAnnotation.tags,
        ).where(HypothesisAnnotation.document_id.is_not(None))

        if group_id:
            stmt = stmt.where(HypothesisAnnotation.group_id == group_id)

        doc_codes: dict[str, dict[str, set[str]]] = {}

        scanned = 0
        for doc_id, tags in db.execute(stmt).yield_per(5000):
            scanned += 1
            if not doc_id:
                continue
            if doc_scope is not None and doc_id not in doc_scope:
                continue

            v1, ext, allc = set(), set(), set()
            # resolve tags -> canonical codes (registry + aliases + fuzzy)
            for raw in (tags or []):
                canonical = resolve_tag_to_canonical(raw, canon_version, alias_to_canon, key_to_canon)
                if not canonical:
                    continue
                ver = canon_version.get(canonical)
                if not ver:
                    continue
                allc.add(canonical)
                if ver == "v1":
                    v1.add(canonical)
                else:
                    ext.add(canonical)

            if not allc:
                continue

            bucket = doc_codes.setdefault(doc_id, {"v1": set(), "ext": set(), "all": set()})
            bucket["v1"].update(v1)
            bucket["ext"].update(ext)
            bucket["all"].update(allc)

        updated = solr_update_codes_only(core, doc_codes)

        return {
            "ok": True,
            "core": core,
            "project_id": str(project_id) if project_id else None,
            "group_id": group_id,
            "annotation_rows_scanned": scanned,
            "docs_with_codes": len(doc_codes),
            "docs_updated_in_solr": updated,
        }
    finally:
        db.close()

@app.post("/solr/recompute_projects")
def recompute_solr_projects(core: str = "hitl_test", project_id: Optional[UUID] = None):
    """
    Recompute Solr project_ids_ss purely from Postgres project_documents.
    If project_id is provided: only that project is applied.
    If not provided: rebuild for all project_documents.
    """
    db = SessionLocal()
    try:
        stmt = select(ProjectDocument.project_id, ProjectDocument.document_id)
        if project_id:
            stmt = stmt.where(ProjectDocument.project_id == project_id)

        rows = db.execute(stmt).all()

        doc_to_projects: dict[str, set[str]] = {}
        for pid, did in rows:
            if not did:
                continue
            bucket = doc_to_projects.setdefault(did, set())
            bucket.add(str(pid))

        updated = solr_set_project_membership(core, doc_to_projects)

        return {
            "ok": True,
            "core": core,
            "project_id": str(project_id) if project_id else None,
            "project_document_rows_scanned": len(rows),
            "docs_updated_in_solr": updated,
        }
    finally:
        db.close()


##Add the minimal code-registry API (optional but recommended)

# If you want users/admins to actually “Add Codes” without touching DB manually.

##
class CodeCreate(BaseModel):
    code: str
    display_name: Optional[str] = None
    description: Optional[str] = None


class AliasCreate(BaseModel):
    alias: str


@app.get("/codes")
def list_codes(include_inactive: bool = False):
    db = SessionLocal()
    try:
        q = select(Code)
        if not include_inactive:
            q = q.where(Code.is_active == True)

        codes = db.execute(q).scalars().all()
        out = []
        for c in codes:
            aliases = db.execute(select(CodeAlias.alias).where(CodeAlias.code == c.code)).scalars().all()
            out.append({
                "code": c.code,
                "version": c.version,
                "display_name": c.display_name,
                "description": c.description,
                "is_active": c.is_active,
                "is_locked": c.is_locked,
                "aliases": aliases,
            })
        return {"codes": out}
    finally:
        db.close()


@app.post("/codes")
def create_code(payload: CodeCreate):
    db = SessionLocal()
    try:
        code = normalize_tag(payload.code)
        if not code:
            raise HTTPException(400, "code is empty")

        if db.get(Code, code):
            raise HTTPException(409, "code already exists")

        row = Code(
            code=code,
            version="ext",
            display_name=payload.display_name,
            description=payload.description,
            is_active=True,
            is_locked=False,
        )
        db.add(row)
        db.commit()
        return {"ok": True, "code": code, "version": "ext"}
    finally:
        db.close()


@app.post("/codes/{code}/aliases")
def add_alias(code: str, payload: AliasCreate):
    db = SessionLocal()
    try:
        code = normalize_tag(code)
        alias = normalize_tag(payload.alias)

        c = db.get(Code, code)
        if not c or not c.is_active:
            raise HTTPException(404, "unknown code")

        if db.get(CodeAlias, alias):
            raise HTTPException(409, "alias already exists")

        db.add(CodeAlias(alias=alias, code=code))
        db.commit()
        return {"ok": True, "code": code, "alias": alias}
    finally:
        db.close()


@app.patch("/codes/{code}/deactivate")
def deactivate_code(code: str):
    db = SessionLocal()
    try:
        code = normalize_tag(code)
        row = db.get(Code, code)
        if not row:
            raise HTTPException(404, "unknown code")
        if row.is_locked:
            raise HTTPException(400, "cannot deactivate locked v1 code")

        row.is_active = False
        db.commit()
        return {"ok": True, "code": code}
    finally:
        db.close()


################ CSV export end point##############
# 4) How to run it
# Export all docs in a project
# curl -L -o out.csv "http://localhost:8000/export/csv?project_id=YOUR_PROJECT_ID"
#
# Export a single document
# curl -L -o out.csv "http://localhost:8000/export/csv?project_id=YOUR_PROJECT_ID&document_id=DOC_ID"
#
# Export selected docs
# curl -L -o out.csv "http://localhost:8000/export/csv?project_id=YOUR_PROJECT_ID&document_ids=DOC1,DOC2,DOC3"
#
# Export only one code (by canonical name or alias/variant)
# curl -L -o out.csv "http://localhost:8000/export/csv?project_id=YOUR_PROJECT_ID&code=Confess/Plead"
#
# Export only v1 codes
# curl -L -o out.csv ("http://localhost:8000/export/csv?project_id=YOUR_PROJECT_ID&version=v1"

################################################################################################
@app.get("/export/csv")
def export_csv(
    project_id: UUID,
    core: str = "hitl_test",
    document_id: Optional[str] = None,
    document_ids: Optional[str] = None,
    code: Optional[str] = None,
    version: str = "all",
    source: str = "human",
    include_annotators: bool = False,
):
    """
    Production-grade long/tidy export:
      one row per (document_id, code)

    Values:
      - value: latest non-empty annotation.text
      - values: JSON array string of all unique non-empty texts
      - has_span + span_examples
    """
    if source not in {"human", "all"}:
        # model not implemented yet
        raise HTTPException(400, "source supports: human|all (model export not implemented yet)")

    if version not in {"v1", "ext", "all"}:
        raise HTTPException(400, "version must be v1|ext|all")

    db = SessionLocal()

    try:
        # 1) determine doc set (project-scoped)
        doc_ids = iter_project_document_ids(db, str(project_id), document_id=document_id, document_ids=document_ids)
        if not doc_ids:
            raise HTTPException(404, "No documents matched (check project membership and document_id(s))")

        # 2) preload doc_id -> canonical_url
        doc_rows = db.execute(
            select(Document.document_id, Document.canonical_url).where(Document.document_id.in_(doc_ids))
        ).all()
        doc_url = {d: u for (d, u) in doc_rows}

        # 3) preload code maps ONCE (fast; avoids DB hits per annotation)
        canon_version, alias_to_canon, key_to_canon = load_code_maps(db)

        # If caller provided `code=...`, normalize it to canonical too
        code_filter: Optional[str] = None
        if code:
            cf = resolve_tag_to_canonical(code, canon_version, alias_to_canon, key_to_canon)
            if not cf:
                # if they asked for an unknown/unregistered code, return empty CSV (header only)
                code_filter = "__NO_MATCH__"
            else:
                code_filter = cf

        # 4) stream CSV
        headers = [
            "project_id",
            "document_id",
            "canonical_url",
            "code",
            "code_version",
            "source",
            "value",
            "value_mode",
            "values",
            "n_values",
            "has_span",
            "span_examples",
            "n_annotations",
            "latest_updated",
        ]
        if include_annotators:
            headers.append("annotators")

        def gen():
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=headers)
            w.writeheader()
            yield buf.getvalue()
            buf.seek(0)
            buf.truncate(0)

            # Aggregation structure:
            # (doc_id, code) -> metrics
            agg: dict[tuple[str, str], dict] = {}

            # Pull annotations for docs in chunks
            stmt = select(
                HypothesisAnnotation.document_id,
                HypothesisAnnotation.tags,
                HypothesisAnnotation.text,
                HypothesisAnnotation.exact,
                HypothesisAnnotation.user,
                HypothesisAnnotation.updated,
            ).where(HypothesisAnnotation.document_id.in_(doc_ids))

            # stream rows from DB
            for doc_id, tags, text, exact, user, updated in db.execute(stmt).yield_per(5000):
                if not doc_id:
                    continue

                resolved = resolve_codes_for_tags_cached(tags or [], canon_version, alias_to_canon, key_to_canon)
                if not resolved:
                    continue

                # normalize the value
                val = (text or "").strip()
                upd = parse_dt_utc(updated)

                for canonical_code, code_ver in resolved:
                    if code_filter and canonical_code != code_filter:
                        continue
                    if version != "all" and code_ver != version:
                        continue

                    key = (doc_id, canonical_code)
                    rec = agg.get(key)
                    if not rec:
                        rec = {
                            "code_version": code_ver,
                            "n_annotations": 0,
                            "has_span": False,
                            "span_examples": [],
                            "values_set": set(),
                            "latest_value": None,      # str
                            "latest_updated": None,    # datetime
                            "annotators": set(),
                        }
                        agg[key] = rec

                    rec["n_annotations"] += 1

                    if user:
                        rec["annotators"].add(user)

                    # span?
                    if exact and str(exact).strip():
                        rec["has_span"] = True
                        # keep up to 3 unique examples
                        ex = str(exact).strip()
                        if ex and ex not in rec["span_examples"] and len(rec["span_examples"]) < 3:
                            rec["span_examples"].append(ex)

                    # values aggregation
                    if val:
                        rec["values_set"].add(val)
                        # pick latest non-empty by updated timestamp
                        if upd:
                            if rec["latest_updated"] is None or upd > rec["latest_updated"]:
                                rec["latest_updated"] = upd
                                rec["latest_value"] = val
                        else:
                            # if no updated, keep first non-empty as fallback
                            if rec["latest_value"] is None:
                                rec["latest_value"] = val

            # Emit rows (sorted for stability)
            for (doc_id, canonical_code) in sorted(agg.keys()):
                rec = agg[(doc_id, canonical_code)]
                values_list = sorted(list(rec["values_set"]))
                values_json = json.dumps(values_list, ensure_ascii=False)

                row = {
                    "project_id": str(project_id),
                    "document_id": doc_id,
                    "canonical_url": doc_url.get(doc_id) or "",
                    "code": canonical_code,
                    "code_version": rec["code_version"],
                    "source": "human",
                    "value": rec["latest_value"] or "",
                    "value_mode": "latest_nonempty_text",
                    "values": values_json,
                    "n_values": len(values_list),
                    "has_span": bool(rec["has_span"]),
                    "span_examples": " || ".join(rec["span_examples"]) if rec["span_examples"] else "",
                    "n_annotations": rec["n_annotations"],
                    "latest_updated": iso_z(rec["latest_updated"]),
                }
                if include_annotators:
                    row["annotators"] = ";".join(sorted(rec["annotators"])) if rec["annotators"] else ""

                w.writerow(row)
                yield buf.getvalue()
                buf.seek(0)
                buf.truncate(0)

        filename = f"export_project_{project_id}.csv"
        return StreamingResponse(
            gen(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    finally:
        db.close()



#
#
#
# docker compose build api && docker compose up -d --force-recreate api
# curl -L -o wide.csv "http://localhost:8000/export/csv_wide?project_id=5749b0e5-a0dd-4266-8a93-a4dd4114ee57"
# head -n 3 wide.csv


@app.get("/export/csv_wide")
def export_csv_wide(
    project_id: UUID,
    document_id: Optional[str] = None,
    document_ids: Optional[str] = None,  # comma-separated
    code: Optional[str] = None,          # filter to one code (alias/variant ok)
    version: str = "all",                # v1|ext|all
    metric: str = "value",               # value|count|binary
):
    if version not in {"v1", "ext", "all"}:
        raise HTTPException(400, "version must be v1|ext|all")
    if metric not in {"value", "count", "binary"}:
        raise HTTPException(400, "metric must be value|count|binary")

    db = SessionLocal()
    try:
        # 1) doc set (project-scoped)
        doc_ids = iter_project_document_ids(db, str(project_id), document_id=document_id, document_ids=document_ids)
        if not doc_ids:
            raise HTTPException(404, "No documents matched (check project membership and document_id(s))")

        # 2) doc_id -> canonical_url
        doc_rows = db.execute(
            select(Document.document_id, Document.canonical_url).where(Document.document_id.in_(doc_ids))
        ).all()
        doc_url = {d: u for (d, u) in doc_rows}

        # 3) load code maps once
        canon_version, alias_to_canon, key_to_canon = load_code_maps(db)

        # optional code filter normalized to canonical
        code_filter: Optional[str] = None
        if code:
            cf = resolve_tag_to_canonical(code, canon_version, alias_to_canon, key_to_canon)
            if not cf:
                # return a header-only CSV with base columns
                cf = "__NO_MATCH__"
            code_filter = cf

        # 4) aggregate across annotations for these docs
        per_doc, codes_seen = build_wide_aggregates(
            db,
            doc_ids,
            canon_version,
            alias_to_canon,
            key_to_canon,
            version=version,
            code_filter=code_filter if code_filter != "__NO_MATCH__" else "__NO_MATCH__",
        )

        # If user asked for an unknown code, produce empty wide body with only base headers
        if code_filter == "__NO_MATCH__":
            codes_seen = set()

        # 5) build column list
        code_cols = sorted([csv_safe_col(c) for c in codes_seen])
        base_cols = ["project_id", "document_id", "canonical_url"]
        headers = base_cols + code_cols

        def gen():
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=headers)
            w.writeheader()
            yield buf.getvalue()
            buf.seek(0)
            buf.truncate(0)

            # For stable iteration:
            codes_sorted = sorted(list(codes_seen))

            for doc_id in sorted(doc_ids):
                row = {
                    "project_id": str(project_id),
                    "document_id": doc_id,
                    "canonical_url": doc_url.get(doc_id) or "",
                }

                doc_bucket = per_doc.get(doc_id, {})

                for canonical_code in codes_sorted:
                    col = csv_safe_col(canonical_code)
                    rec = doc_bucket.get(canonical_code)

                    if not rec:
                        row[col] = "" if metric == "value" else (0 if metric in {"count", "binary"} else "")
                        continue

                    if metric == "value":
                        row[col] = rec["latest_value"] or ""
                    elif metric == "count":
                        row[col] = rec["count"]
                    else:  # binary
                        row[col] = 1

                w.writerow(row)
                yield buf.getvalue()
                buf.seek(0)
                buf.truncate(0)

        filename = f"export_wide_project_{str(project_id)}.csv"
        return StreamingResponse(
            gen(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    finally:
        db.close()


###############
# Project and teams apis

# POST /teams (bootstrap teams properly)
# # POST /projects (create project)
# # GET /projects (list projects)
# # GET /projects/{project_id} (basic stats)
# # POST /projects/{project_id}/documents/add (add list of document_ids)
# # GET /projects/{project_id}/documents (paginate membership)
#
# And we will also update Solr membership project_ids_ss via atomic updates (so Solr filters work).

#################################################################################

@app.post("/teams")
def create_team(payload: TeamCreateRequest):
    db = SessionLocal()
    try:
        t = Team(team_id=uuid.uuid4(), name=payload.name.strip())
        db.add(t)
        db.commit()
        return {"ok": True, "team_id": str(t.team_id), "name": t.name}
    finally:
        db.close()


@app.post("/projects")
def create_project(payload: ProjectCreateRequest):
    db = SessionLocal()
    try:
        team = db.get(Team, payload.team_id)
        if not team:
            raise HTTPException(404, "team not found")

        p = Project(project_id=uuid.uuid4(), team_id=payload.team_id, name=payload.name.strip())
        db.add(p)
        db.commit()
        return {"ok": True, "project_id": str(p.project_id), "team_id": str(p.team_id), "name": p.name}
    finally:
        db.close()


@app.get("/projects")
def list_projects(team_id: UUID | None = None):
    db = SessionLocal()
    try:
        q = select(Project)
        if team_id:
            q = q.where(Project.team_id == team_id)
        projects = db.execute(q).scalars().all()
        return {
            "projects": [
                {"project_id": str(p.project_id), "team_id": str(p.team_id), "name": p.name}
                for p in projects
            ]
        }
    finally:
        db.close()


@app.get("/projects/{project_id}")
def get_project(project_id: UUID):
    db = SessionLocal()
    try:
        p = db.get(Project, project_id)
        if not p:
            raise HTTPException(404, "project not found")

        n_docs = db.execute(
            select(func.count()).select_from(ProjectDocument).where(ProjectDocument.project_id == project_id)
        ).scalar_one()

        n_docs_with_ann = db.execute(
            select(func.count(func.distinct(HypothesisAnnotation.document_id)))
            .select_from(HypothesisAnnotation)
            .join(ProjectDocument, ProjectDocument.document_id == HypothesisAnnotation.document_id)
            .where(ProjectDocument.project_id == project_id)
        ).scalar_one()

        return {
            "project_id": str(p.project_id),
            "team_id": str(p.team_id),
            "name": p.name,
            "documents_total": int(n_docs),
            "documents_with_human_annotations": int(n_docs_with_ann),
        }
    finally:
        db.close()


@app.post("/projects/{project_id}/documents/add")
def add_documents_to_project(
    project_id: UUID,
    payload: ProjectAddDocsRequest,
    core: str = "hitl_test",
):
    db = SessionLocal()
    try:
        p = db.get(Project, project_id)
        if not p:
            raise HTTPException(404, "project not found")

        existing = set(
            db.execute(select(Document.document_id).where(Document.document_id.in_(payload.document_ids))).scalars().all()
        )
        missing = [d for d in payload.document_ids if d not in existing]
        if missing:
            raise HTTPException(400, f"{len(missing)} document_ids not found")

        added = 0
        for did in payload.document_ids:
            row = db.get(ProjectDocument, {"project_id": project_id, "document_id": did})
            if row:
                continue
            db.add(ProjectDocument(project_id=project_id, document_id=did))
            added += 1
        db.commit()

        solr_updated = solr_add_project_membership(core, str(project_id), payload.document_ids)
        return {"ok": True, "project_id": str(project_id), "docs_added": added, "solr_docs_updated": solr_updated}
    finally:
        db.close()


@app.get("/projects/{project_id}/documents")
def list_project_documents(project_id: UUID, limit: int = 50, offset: int = 0):
    db = SessionLocal()
    try:
        p = db.get(Project, project_id)
        if not p:
            raise HTTPException(404, "project not found")

        rows = db.execute(
            select(ProjectDocument.document_id)
            .where(ProjectDocument.project_id == project_id)
            .order_by(ProjectDocument.document_id)
            .limit(limit)
            .offset(offset)
        ).scalars().all()

        return {"project_id": str(project_id), "document_ids": rows, "limit": limit, "offset": offset}
    finally:
        db.close()




############# SEARCH and SAMPLE ###########
# GET /search endpoint (facets + filtering + project scope)
# What it supports
#
# q free-text (default *:*)
#
# fq repeatable filters
#
# project_id → auto adds fq=project_ids_ss:<uuid>
#
# facets on:
# doc_type_s, source_s, judges_ss, has_human_b, codes_all_ss, appeal_outcome_s
#
# pagination: start, rows
#
# returns: docs + facets + numFound

@app.get("/search")
def search(
    core: str = "hitl_test",
    q: str = "*:*",
    fq: Optional[list[str]] = None,      # repeat ?fq=...&fq=...
    project_id: Optional[UUID] = None,
    rows: int = 20,
    start: int = 0,
    facet: bool = True,
):
    fq_list = normalize_fq_list(fq)

    if project_id:
        fq_list.append(f"project_ids_ss:{solr_escape_term(str(project_id))}")

    facet_fields = [
        "doc_type_s",
        "source_s",
        "judges_ss",
        "has_human_b",
        "codes_all_ss",
        "appeal_outcome_s",
    ]

    params = {
        "q": q or "*:*",
        "rows": rows,
        "start": start,
        "wt": "json",
        "fl": ",".join([
            "document_id_s",
            "canonical_url_s",
            "title_txt",
            "excerpt_txt",
            "published_dt",
            "doc_type_s",
            "source_s",
            "judges_ss",
            "has_human_b",
            "has_any_span_b",
            "codes_v1_ss",
            "codes_ext_ss",
            "codes_all_ss",
            "project_ids_ss",
        ]),
    }

    if fq_list:
        params["fq"] = fq_list

    if facet:
        params.update({
            "facet": "true",
            "facet.mincount": 1,
            "facet.limit": 50,
        })
        # multiple facet.field supported by repeating param in requests
        # easiest: add as list
        params["facet.field"] = facet_fields

    data = solr_select(core, params)

    resp = data.get("response", {}) or {}
    out = {
        "ok": True,
        "core": core,
        "q": q,
        "fq": fq_list,
        "numFound": resp.get("numFound", 0),
        "start": resp.get("start", 0),
        "docs": resp.get("docs", []) or [],
    }

    if facet:
        out["facets"] = data.get("facet_counts", {}) or {}

    return out

# 3) POST /sample endpoint (random sample via rand_f)
# Why this is the right approach
#
# Because you have rand_f stored and indexed, we can sample by:
#
# generate a random float r in [0,1)
#
# query rand_f:[r TO 1] with base filters → get n docs
#
# if insufficient, wrap: rand_f:[0 TO r) to fill remaining
#
# This is fast, scalable, and doesn’t require expensive random sort.
#
# Payload model
class SampleRequest(BaseModel):
    core: str = "hitl_test"
    q: str = "*:*"
    fq: list[str] = Field(default_factory=list)
    project_id: Optional[UUID] = None
    n: int = 20

# Endpoint
@app.post("/sample")
def sample_docs(payload: SampleRequest):
    core = payload.core
    q = payload.q or "*:*"
    n = max(1, min(int(payload.n), 500))  # cap for safety

    fq_list = list(payload.fq or [])
    if payload.project_id:
        fq_list.append(f"project_ids_ss:{solr_escape_term(str(payload.project_id))}")

    # base fields returned
    fl = ",".join([
        "document_id_s",
        "canonical_url_s",
        "title_txt",
        "excerpt_txt",
        "published_dt",
        "doc_type_s",
        "source_s",
        "judges_ss",
        "has_human_b",
        "has_any_span_b",
        "codes_all_ss",
        "project_ids_ss",
        "rand_f",
    ])

    r = random.random()
    fq_hi = fq_list + [f"rand_f:[{r} TO 1]"]
    fq_lo = fq_list + [f"rand_f:[0 TO {r})"]

    def fetch(fqs: list[str], need: int) -> list[dict]:
        if need <= 0:
            return []
        params = {
            "q": q,
            "fq": fqs,
            "rows": need,
            "start": 0,
            "wt": "json",
            "fl": fl,
        }
        data = solr_select(core, params)
        return (data.get("response", {}) or {}).get("docs", []) or []

    docs = fetch(fq_hi, n)
    if len(docs) < n:
        more = fetch(fq_lo, n - len(docs))
        docs.extend(more)

    # de-dupe by document_id_s (just in case)
    seen = set()
    uniq = []
    for d in docs:
        did = d.get("document_id_s")
        if not did or did in seen:
            continue
        seen.add(did)
        uniq.append(d)
        if len(uniq) >= n:
            break

    return {
        "ok": True,
        "core": core,
        "q": q,
        "fq": fq_list,
        "n_requested": n,
        "n_returned": len(uniq),
        "rand_seed": r,
        "docs": uniq,
    }
