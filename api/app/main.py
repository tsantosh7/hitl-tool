# api/app/main.py
import os
import json
import random
import requests
import urllib.parse
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Iterable, Tuple
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, HttpUrl
from sqlalchemy import select

import queue
import threading


from .db import SessionLocal
from .init_db import init
from .models import (
    Team, Project, Document, ProjectDocument,
    HypothesisGroup, HypothesisAnnotation
)

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
# Annotations helpers
# ------------------------------------------------------------------------------

V1_CODES_PATH = os.getenv("V1_CODES_PATH", "/app/data/schema_v1_codes.json")

def load_v1_codes() -> set[str]:
    try:
        with open(V1_CODES_PATH, "r", encoding="utf-8") as f:
            obj = json.load(f)
        codes = obj.get("v1_codes") or []
        return {str(c).strip() for c in codes if str(c).strip()}
    except Exception:
        return set()

V1_CODES: set[str] = load_v1_codes()

def normalize_code(tag: str) -> str:
    # Conservative: do not change semantics, only strip whitespace.
    return (tag or "").strip()

def split_codes(tags: list[str]) -> tuple[set[str], set[str], set[str]]:
    v1: set[str] = set()
    ext: set[str] = set()
    allc: set[str] = set()
    for t in tags or []:
        c = normalize_code(t)
        if not c:
            continue
        allc.add(c)
        if c in V1_CODES:
            v1.add(c)
        else:
            ext.add(c)
    return v1, ext, allc

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


# def upsert_annotations_bulk(
#     db,
#     fields_list: List[dict],
#     url_to_doc: Dict[str, str],
# ) -> Tuple[int, int, Dict[str, Dict[str, bool]]]:
#     """
#     Upsert annotations. Returns:
#       (annotations_seen, annotations_linked_to_docs, doc_flags_for_solr)
#     """
#     seen = 0
#     linked = 0
#     doc_flags: Dict[str, Dict[str, bool]] = {}
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
#             if doc_id not in doc_flags:
#                 doc_flags[doc_id] = {"has_human": True, "has_any_span": False}
#             if fields.get("exact") and str(fields.get("exact")).strip():
#                 doc_flags[doc_id]["has_any_span"] = True
#
#     return seen, linked, doc_flags

from typing import DefaultDict
from collections import defaultdict

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

        row = db.get(HypothesisAnnotation, ann_id)
        if row:
            new_updated = fields.get("updated")
            if row.updated and new_updated and new_updated <= row.updated:
                pass
            else:
                row.group_id = fields["group_id"]
                row.document_id = doc_id
                row.canonical_url = canon
                row.user = fields.get("user")
                row.created = fields.get("created")
                row.updated = fields.get("updated")
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
                created=fields.get("created"),
                updated=fields.get("updated"),
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

            # Codes
            tags = fields.get("tags") or []
            v1, ext, allc = split_codes(tags)
            bucket = doc_codes.setdefault(doc_id, {"v1": set(), "ext": set(), "all": set()})
            bucket["v1"].update(v1)
            bucket["ext"].update(ext)
            bucket["all"].update(allc)

    return seen, linked, doc_flags, doc_codes

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

            if g_row and not payload.force_full and max_updated_str:
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
