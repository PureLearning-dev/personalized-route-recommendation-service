"""服务运行配置。"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从环境变量和可选 .env 文件读取配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["development", "test", "production"] = "development"
    database_url: str = (
        "postgresql+psycopg://personalized:personalized@localhost:5432/"
        "personalized_recommendations_core"
    )
    model_cost_unit: str = "CNY"
    log_level: str = "INFO"

    @model_validator(mode="after")
    def validate_production_secrets(self) -> Settings:
        if not self.model_cost_unit.strip():
            raise ValueError("MODEL_COST_UNIT 不能为空")
        if self.app_env == "production" and "localhost" in self.database_url:
            raise ValueError("生产环境 DATABASE_URL 不能使用默认本地地址")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
