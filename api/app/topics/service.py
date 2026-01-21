# api/app/topics/service.py
from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ..models import TopicRun


def get_or_create_active_global_run(db: Session, *, actor: str | None = None) -> TopicRun:
    """
    Returns the active global TopicRun (project_id is NULL).
    If none is active:
      - activates the most recently created global run, OR
      - creates a new global run and marks it active.
    """
    run = db.execute(
        select(TopicRun)
        .where(TopicRun.project_id.is_(None), TopicRun.is_active.is_(True))
        .limit(1)
    ).scalar_one_or_none()

    if run:
        return run

    latest = db.execute(
        select(TopicRun)
        .where(TopicRun.project_id.is_(None))
        .order_by(TopicRun.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()

    # ensure only one global run active
    db.execute(
        update(TopicRun)
        .where(TopicRun.project_id.is_(None))
        .values(is_active=False)
    )

    if latest:
        latest.is_active = True
        db.add(latest)
        db.commit()
        db.refresh(latest)
        return latest

    run = TopicRun(
        project_id=None,
        name="topics_global_v1",
        topic_schema_version="topics-v1",
        method="human+propagation",
        is_active=True,
        created_by=actor,
        params={},
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run
