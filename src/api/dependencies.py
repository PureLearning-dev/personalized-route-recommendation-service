"""FastAPI 请求级依赖。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends

from ..config import Settings, get_settings
from ..infrastructure.db.repositories import SqlAlchemyRepository
from ..infrastructure.db.session import get_session_factory


def repository_dependency() -> Iterator[SqlAlchemyRepository]:
    factory = get_session_factory()
    with factory() as session:
        repository = SqlAlchemyRepository(session)
        try:
            yield repository
        finally:
            if session.in_transaction():
                session.rollback()


Repository = Annotated[SqlAlchemyRepository, Depends(repository_dependency)]
SettingsDependency = Annotated[Settings, Depends(get_settings)]
