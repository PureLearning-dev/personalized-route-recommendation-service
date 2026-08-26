"""应用服务对外暴露的稳定错误类型。"""

from __future__ import annotations

from typing import Any


class ApplicationError(RuntimeError):
    status_code = 400
    code = "APPLICATION_ERROR"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(ApplicationError):
    status_code = 404
    code = "RESOURCE_NOT_FOUND"


class ConflictError(ApplicationError):
    status_code = 409
    code = "RESOURCE_CONFLICT"


class ValidationError(ApplicationError):
    status_code = 422
    code = "DOMAIN_VALIDATION_ERROR"


class ServiceUnavailableError(ApplicationError):
    status_code = 503
    code = "SERVICE_UNAVAILABLE"
