from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorBody(ApiModel):
    code: str
    message: str
    details: dict[str, Any]
    request_id: str


class ErrorResponse(ApiModel):
    error: ErrorBody
