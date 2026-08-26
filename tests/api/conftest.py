from __future__ import annotations

# 测试数据库环境必须先设置，再导入会缓存配置的应用模块。
# ruff: noqa: E402
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://personalized:personalized@localhost:5432/"
    "personalized_recommendations_core_test",
)
os.environ["APP_ENV"] = "test"
os.environ["DATABASE_URL"] = TEST_DATABASE_URL

from src.api.main import create_app
from src.config import get_settings
from src.infrastructure.db.session import get_engine, reset_database_singletons


@dataclass(frozen=True)
class ApiHarness:
    client: TestClient


@pytest.fixture(scope="session", autouse=True)
def migrated_database() -> Iterator[None]:
    get_settings.cache_clear()
    reset_database_singletons()
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    command.upgrade(config, "head")
    yield
    reset_database_singletons()


@pytest.fixture(autouse=True)
def clean_database(migrated_database: None) -> Iterator[None]:
    with get_engine().begin() as connection:
        connection.execute(text("TRUNCATE TABLE users CASCADE"))
    yield


def _new_harness() -> ApiHarness:
    return ApiHarness(client=TestClient(create_app()))


@pytest.fixture
def api() -> Iterator[ApiHarness]:
    harness = _new_harness()
    with harness.client:
        yield harness
