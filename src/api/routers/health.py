from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from ..dependencies import Repository

router = APIRouter(tags=["health"])


@lru_cache(maxsize=1)
def _expected_migration_head() -> str:
    project_root = Path(__file__).resolve().parents[3]
    config = Config(str(project_root / "alembic.ini"))
    config.set_main_option("script_location", str(project_root / "migrations"))
    head = ScriptDirectory.from_config(config).get_current_head()
    if head is None:
        raise RuntimeError("未找到 Alembic migration head")
    return head


@router.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready", response_model=None)
def ready(repository: Repository) -> dict[str, str] | JSONResponse:
    try:
        repository.session.execute(text("SELECT 1"))
        revision = repository.session.scalar(text("SELECT version_num FROM alembic_version"))
        if revision != _expected_migration_head():
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "migration_not_current"},
            )
    except (SQLAlchemyError, RuntimeError):
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "database_unavailable"},
        )
    return {"status": "ready"}
