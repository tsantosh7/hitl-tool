# api/app/ui/routes.py
from __future__ import annotations

from fastapi import APIRouter, Request, Depends, Form
from fastapi.responses import HTMLResponse, RedirectResponse
import httpx
import urllib.parse

from app.auth.deps import require_user, require_role

router = APIRouter(prefix="/ui", tags=["ui"])


# -------------------------
# Internal ASGI calls to your API endpoints (no network hop)
# -------------------------
async def asgi_get(request: Request, path: str, params: dict | None = None):
    transport = httpx.ASGITransport(app=request.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://app") as client:
        r = await client.get(path, params=params)
        r.raise_for_status()
        return r.json()


async def asgi_post_json(request: Request, path: str, json_body: dict):
    transport = httpx.ASGITransport(app=request.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://app") as client:
        r = await client.post(path, json=json_body)
        r.raise_for_status()
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        return {"ok": True}


async def asgi_patch(request: Request, path: str):
    transport = httpx.ASGITransport(app=request.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://app") as client:
        r = await client.patch(path)
        r.raise_for_status()
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        return {"ok": True}


def build_hypothesis_incontext(url: str, group_id: str = "__world__") -> str:
    return (
        "https://hyp.is/go?url="
        + urllib.parse.quote(url, safe="")
        + "&group="
        + urllib.parse.quote(group_id, safe="")
    )


def build_hypothesis_direct(url: str, group_id: str = "__world__") -> str:
    return (
        "https://hypothes.is/?url="
        + urllib.parse.quote(url, safe="")
        + "&group="
        + urllib.parse.quote(group_id, safe="")
    )


# -------------------------
# Session helpers
# -------------------------
def _get_project_id(request: Request) -> str | None:
    return request.session.get("project_id")


def _set_project_id(request: Request, project_id: str) -> None:
    request.session["project_id"] = project_id


def _get_project_name(request: Request) -> str | None:
    return request.session.get("project_name")


def _set_project_name(request: Request, name: str | None) -> None:
    if name:
        request.session["project_name"] = name
    else:
        request.session.pop("project_name", None)


def _get_run_id(request: Request) -> str | None:
    return request.session.get("topic_run_id")


def _set_run_id(request: Request, run_id: str | None) -> None:
    if run_id:
        request.session["topic_run_id"] = run_id
    else:
        request.session.pop("topic_run_id", None)


async def _ensure_project_selected(request: Request) -> str | None:
    return _get_project_id(request)


async def _pick_run_id_for_project(request: Request, project_id: str) -> str | None:
    runs_res = await asgi_get(request, "/topics/runs", params={"project_id": project_id})
    runs = runs_res.get("runs", []) or []
    if not runs:
        return None
    active = next((r for r in runs if r.get("is_active")), None)
    chosen = active or runs[0]
    return chosen.get("run_id")


# -------------------------
# Friendly keyword -> Solr query builder
# -------------------------
def _solr_escape_phrase(s: str) -> str:
    """Safe for Solr phrase queries: field:"...". Escapes backslash + quotes."""
    s = (s or "").strip()
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return s


def _solr_escape_query(s: str) -> str:
    """
    Safe-ish escape for edismax query text. We escape common special chars so user
    can type normal text without breaking Solr parser.
    """
    s = (s or "").strip()
    if not s:
        return ""
    for ch in r'+-!():^[]"{}~*?|&\\/':
        s = s.replace(ch, "\\" + ch)
    return s


def build_user_friendly_q(kw: str | None, kw_field: str, include_codes_topics: bool) -> str:
    """
    User-friendly search:
      - title/excerpt/body full text
      - human/model extracted values (raw + normalized)
    Optional:
      - codes_all_ss, topics_ss, topic_keys_ss, topic_kv_ss

    kw_field:
      all      -> broad search
      title    -> title only
      excerpt  -> excerpt only
      body     -> body only
      values   -> values_* only
      url      -> canonical_url_s exact
      id       -> document_id_s exact
    """
    kw = (kw or "").strip()
    if not kw:
        return "*:*"

    if kw_field == "url":
        return f'canonical_url_s:"{_solr_escape_phrase(kw)}"'
    if kw_field == "id":
        return f'document_id_s:"{_solr_escape_phrase(kw)}"'

    qtext = _solr_escape_query(kw)

    # Text fields (edismax qf)
    qf_all = (
        "title_txt^4 "
        "excerpt_txt^2 "
        "body_txt "
        "values_human_txt "
        "values_model_txt "
        "values_human_norm_txt "
        "values_model_norm_txt"
    )
    qf_title = "title_txt^4"
    qf_excerpt = "excerpt_txt^2"
    qf_body = "body_txt"
    qf_values = "values_human_txt values_model_txt values_human_norm_txt values_model_norm_txt"

    if kw_field == "title":
        qf = qf_title
    elif kw_field == "excerpt":
        qf = qf_excerpt
    elif kw_field == "body":
        qf = qf_body
    elif kw_field == "values":
        qf = qf_values
    else:
        qf = qf_all

    base = f'{{!edismax qf="{qf}" pf="{qf_title}" mm=1}}{qtext}'

    if include_codes_topics:
        p = _solr_escape_phrase(kw)
        extra = (
            f' OR codes_all_ss:"{p}"'
            f' OR topics_ss:"{p}"'
            f' OR topic_keys_ss:"{p}"'
            f' OR topic_kv_ss:"{p}"'
        )
        return f"({base}{extra})"

    return base


# -------------------------
# Root -> Dashboard
# -------------------------
@router.get("/", response_class=HTMLResponse)
def ui_root(request: Request):
    return RedirectResponse("/ui/dashboard", status_code=303)


# -------------------------
# Dashboard
# -------------------------
@router.get("/dashboard", response_class=HTMLResponse)
async def ui_dashboard(request: Request, user=Depends(require_user)):
    projects_res = await asgi_get(request, "/projects", params={})
    projects = projects_res.get("projects", []) or []
    selected_project_id = _get_project_id(request)
    selected_project_name = _get_project_name(request)

    return request.app.state.templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": user,
            "projects": projects,
            "has_projects": bool(projects),
            "selected_project_id": selected_project_id,
            "selected_project_name": selected_project_name,
        },
    )


@router.post("/select_project")
async def ui_select_project(
    request: Request,
    project_id: str = Form(...),
    user=Depends(require_user),
):
    proj = await asgi_get(request, f"/projects/{project_id}", params={})
    _set_project_id(request, project_id)
    _set_project_name(request, proj.get("name"))
    _set_run_id(request, None)
    return RedirectResponse("/ui/dashboard", status_code=303)


# -------------------------
# Search Documents
# -------------------------
@router.get("/search", response_class=HTMLResponse)
async def ui_search(
    request: Request,
    # Advanced Solr query (optional)
    q: str = "",
    # Friendly keyword search (recommended)
    kw: str | None = None,
    kw_field: str = "all",  # all | title | excerpt | body | values | url | id
    include_codes_topics: str | None = "1",  # checkbox; default ON
    rows: int = 20,
    start: int = 0,
    scope: str = "all",  # all | project
    code: str | None = None,
    topic: str | None = None,
    has_human: str | None = None,
    has_any_span: str | None = None,
    user=Depends(require_user),
):
    project_id = _get_project_id(request)
    project_name = _get_project_name(request)

    fq: list[str] = []
    if code:
        fq.append(f'codes_all_ss:"{_solr_escape_phrase(code)}"')
    if topic:
        fq.append(f'topics_ss:"{_solr_escape_phrase(topic)}"')
    if has_human == "1":
        fq.append("has_human_b:true")
    if has_any_span == "1":
        fq.append("has_any_span_b:true")

    advanced_q = (q or "").strip()
    kw_clean = (kw or "").strip()

    if advanced_q:
        effective_q = advanced_q
    else:
        effective_q = build_user_friendly_q(
            kw_clean,
            kw_field,
            include_codes_topics=(include_codes_topics == "1" or include_codes_topics is None),
        )

    params = {
        "q": effective_q,
        "core": "hitl_test",
        "rows": rows,
        "start": start,
        "fq": fq,
    }

    if scope == "project":
        if not project_id:
            return RedirectResponse("/ui/dashboard", status_code=303)
        params["project_id"] = project_id

    result = await asgi_get(request, "/search", params=params)
    facets = result.get("facets", {}) or {}

    return request.app.state.templates.TemplateResponse(
        "search.html",
        {
            "request": request,
            "user": user,
            "project_id": project_id,
            "project_name": project_name,
            "scope": scope,
            "q": advanced_q,
            "kw": kw_clean,
            "kw_field": kw_field,
            "include_codes_topics": include_codes_topics or "1",
            "rows": rows,
            "start": start,
            "code": code,
            "topic": topic,
            "has_human": has_human,
            "has_any_span": has_any_span,
            "result": result,
            "facets": facets,
        },
    )


# -------------------------
# Add to Project (bulk + from search)
# -------------------------
@router.get("/add_to_project", response_class=HTMLResponse)
async def ui_add_to_project_page(request: Request, user=Depends(require_user)):
    project_id = await _ensure_project_selected(request)
    if not project_id:
        return RedirectResponse("/ui/dashboard", status_code=303)

    return request.app.state.templates.TemplateResponse(
        "add_to_project.html",
        {
            "request": request,
            "user": user,
            "project_id": project_id,
            "project_name": _get_project_name(request),
            "message": None,
        },
    )


@router.post("/add_to_project")
async def ui_add_to_project_post(
    request: Request,
    document_ids_text: str = Form(""),
    user=Depends(require_role("admin", "reviewer")),
):
    project_id = await _ensure_project_selected(request)
    if not project_id:
        return RedirectResponse("/ui/dashboard", status_code=303)

    ids = [x.strip() for x in (document_ids_text or "").splitlines() if x.strip()]
    if not ids:
        return request.app.state.templates.TemplateResponse(
            "add_to_project.html",
            {
                "request": request,
                "user": user,
                "project_id": project_id,
                "project_name": _get_project_name(request),
                "message": "No document IDs provided.",
            },
            status_code=400,
        )

    res = await asgi_post_json(
        request,
        f"/projects/{project_id}/documents/add",
        {"document_ids": ids},
    )

    msg = f"Added: {res.get('docs_added', 0)} | Solr updated: {res.get('solr_docs_updated', 0)}"
    return request.app.state.templates.TemplateResponse(
        "add_to_project.html",
        {
            "request": request,
            "user": user,
            "project_id": project_id,
            "project_name": _get_project_name(request),
            "message": msg,
        },
    )


@router.post("/projects/add_one")
async def ui_add_one_from_search(
    request: Request,
    document_id: str = Form(...),
    user=Depends(require_role("admin", "reviewer")),
):
    project_id = await _ensure_project_selected(request)
    if not project_id:
        return RedirectResponse("/ui/dashboard", status_code=303)

    await asgi_post_json(
        request,
        f"/projects/{project_id}/documents/add",
        {"document_ids": [document_id]},
    )
    return RedirectResponse("/ui/search", status_code=303)


# -------------------------
# Export page
# -------------------------
@router.get("/export", response_class=HTMLResponse)
async def ui_export_page(
    request: Request,
    version: str = "all",
    code: str | None = None,
    include_annotators: str | None = None,
    metric: str = "value",
    user=Depends(require_user),
):
    project_id = await _ensure_project_selected(request)
    if not project_id:
        return RedirectResponse("/ui/dashboard", status_code=303)

    codes_res = await asgi_get(request, "/codes", params={})
    codes = codes_res.get("codes", []) or []

    return request.app.state.templates.TemplateResponse(
        "export.html",
        {
            "request": request,
            "user": user,
            "project_id": project_id,
            "project_name": _get_project_name(request),
            "codes": codes,
            "version": version,
            "code": code,
            "include_annotators": include_annotators,
            "metric": metric,
        },
    )


# -------------------------
# Codes page
# -------------------------
@router.get("/codes", response_class=HTMLResponse)
async def ui_codes_page(
    request: Request,
    include_inactive: str | None = None,
    user=Depends(require_user),
):
    res = await asgi_get(request, "/codes", params={"include_inactive": bool(include_inactive)})
    return request.app.state.templates.TemplateResponse(
        "codes.html",
        {
            "request": request,
            "user": user,
            "codes": res.get("codes", []) or [],
            "include_inactive": include_inactive,
            "message": None,
        },
    )


@router.post("/codes/create")
async def ui_codes_create(
    request: Request,
    code: str = Form(...),
    display_name: str = Form(""),
    description: str = Form(""),
    user=Depends(require_role("admin")),
):
    await asgi_post_json(
        request,
        "/codes",
        {
            "code": code,
            "display_name": display_name or None,
            "description": description or None,
        },
    )
    return RedirectResponse("/ui/codes", status_code=303)


@router.post("/codes/add_alias")
async def ui_codes_add_alias(
    request: Request,
    code: str = Form(...),
    alias: str = Form(...),
    user=Depends(require_role("admin")),
):
    await asgi_post_json(request, f"/codes/{code}/aliases", {"alias": alias})
    return RedirectResponse("/ui/codes", status_code=303)


@router.post("/codes/deactivate")
async def ui_codes_deactivate(
    request: Request,
    code: str = Form(...),
    user=Depends(require_role("admin")),
):
    await asgi_patch(request, f"/codes/{code}/deactivate")
    return RedirectResponse("/ui/codes", status_code=303)


# -------------------------
# Document detail (Review)
# -------------------------
@router.get("/docs/{document_id}", response_class=HTMLResponse)
async def ui_doc_detail(request: Request, document_id: str, user=Depends(require_user)):
    project_id = await _ensure_project_selected(request)
    if not project_id:
        return RedirectResponse("/ui/dashboard", status_code=303)

    in_project = False
    try:
        proj_res = await asgi_get(
            request,
            f"/projects/{project_id}/documents",
            params={"limit": 5000, "offset": 0},
        )
        ids = set(proj_res.get("document_ids", []) or [])
        in_project = document_id in ids
    except Exception:
        in_project = False

    doc_res = await asgi_get(
        request,
        "/search",
        params={
            "q": f'document_id_s:"{_solr_escape_phrase(document_id)}"',
            "core": "hitl_test",
            "rows": 1,
            "start": 0,
        },
    )

    docs = doc_res.get("docs", []) or []
    doc = docs[0] if docs else {"document_id_s": document_id, "title_txt": ["(not found in Solr)"]}

    hypo = None
    try:
        hypo = await asgi_get(request, "/hypothesis/link", params={"document_id": document_id})
    except Exception:
        hypo = None

    if not hypo:
        cu = doc.get("canonical_url_s")
        if isinstance(cu, list):
            cu = cu[0] if cu else None
        if isinstance(cu, str) and cu:
            hypo = {
                "hypothesis_incontext": build_hypothesis_incontext(cu, "__world__"),
                "hypothesis_direct": build_hypothesis_direct(cu, "__world__"),
            }

    run_id = _get_run_id(request)
    if not run_id:
        run_id = await _pick_run_id_for_project(request, project_id)
        _set_run_id(request, run_id)

    topics: list[dict] = []
    if run_id:
        topics_res = await asgi_get(
            request,
            f"/documents/{document_id}/topics",
            params={"run_id": run_id},
        )
        topics = topics_res.get("topics", []) or []

    back_url = request.headers.get("referer") or "/ui/search"

    return request.app.state.templates.TemplateResponse(
        "doc_detail.html",
        {
            "request": request,
            "user": user,
            "project_id": project_id,
            "project_name": _get_project_name(request),
            "doc": doc,
            "hypo": hypo,
            "run_id": run_id,
            "topics": topics,
            "in_project": in_project,
            "back_url": back_url,
        },
    )


# -------------------------
# Topic actions (HTML forms)
# -------------------------
@router.post("/topics/accept")
async def ui_topic_accept(
    request: Request,
    document_id: str = Form(...),
    topic_key: str = Form(...),
    topic_label: str = Form(...),
    run_id: str = Form(...),
    user=Depends(require_role("admin", "reviewer")),
):
    await asgi_post_json(
        request,
        "/topics/label",
        {
            "run_id": run_id,
            "document_id": document_id,
            "topic_key": topic_key,
            "topic_label": topic_label,
            "user": user["username"],
        },
    )
    return RedirectResponse(f"/ui/docs/{document_id}", status_code=303)


@router.post("/topics/reject")
async def ui_topic_reject(
    request: Request,
    document_id: str = Form(...),
    topic_key: str = Form(...),
    run_id: str = Form(...),
    user=Depends(require_role("admin", "reviewer")),
):
    await asgi_post_json(
        request,
        "/topics/reject",
        {
            "run_id": run_id,
            "document_id": document_id,
            "topic_key": topic_key,
            "user": user["username"],
        },
    )
    return RedirectResponse(f"/ui/docs/{document_id}", status_code=303)


@router.post("/topics/delete")
async def ui_topic_delete(
    request: Request,
    document_id: str = Form(...),
    topic_key: str = Form(...),
    run_id: str = Form(...),
    user=Depends(require_role("admin", "reviewer")),
):
    await asgi_post_json(
        request,
        "/topics/label/delete",
        {
            "run_id": run_id,
            "document_id": document_id,
            "topic_key": topic_key,
            "user": user["username"],
        },
    )
    return RedirectResponse(f"/ui/docs/{document_id}", status_code=303)


# -------------------------
# Project creation (bootstrap)
# -------------------------
@router.post("/projects/create")
async def ui_create_project(
    request: Request,
    name: str = Form(...),
    team_name: str = Form("Default Team"),
    user=Depends(require_role("admin", "reviewer")),
):
    created = await asgi_post_json(
        request,
        "/projects/bootstrap",
        {"name": name, "team_name": team_name or "Default Team"},
    )

    project_id = created.get("project_id")
    if project_id:
        _set_project_id(request, project_id)
        _set_project_name(request, name)
        _set_run_id(request, None)

    return RedirectResponse("/ui/search", status_code=303)
