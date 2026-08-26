"""统一错误响应与请求 ID。"""

from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError, OperationalError

from ..application.errors import ApplicationError
from ..profile.exceptions import ProfileNumericalError, ProfileValidationError
from ..recommendation.exceptions import RecommendationValidationError

logger = logging.getLogger(__name__)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", str(uuid4()))


def _response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    details: dict | list | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": details or {},
                "request_id": _request_id(request),
            }
        },
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApplicationError)
    async def application_error(request: Request, error: ApplicationError) -> JSONResponse:
        return _response(
            request,
            status_code=error.status_code,
            code=error.code,
            message=error.message,
            details=error.details,
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation(request: Request, error: RequestValidationError) -> JSONResponse:
        return _response(
            request,
            status_code=422,
            code="REQUEST_VALIDATION_ERROR",
            message="请求数据不符合接口契约",
            details=jsonable_encoder(error.errors()),
        )

    @app.exception_handler(ProfileValidationError)
    @app.exception_handler(RecommendationValidationError)
    async def domain_validation(request: Request, error: Exception) -> JSONResponse:
        return _response(
            request,
            status_code=422,
            code="DOMAIN_VALIDATION_ERROR",
            message=str(error),
        )

    @app.exception_handler(ProfileNumericalError)
    async def numerical_error(request: Request, error: Exception) -> JSONResponse:
        logger.exception("Profile numerical failure")
        return _response(
            request,
            status_code=503,
            code="PROFILE_NUMERICAL_FAILURE",
            message="画像计算暂时失败，请稍后重试",
        )

    @app.exception_handler(IntegrityError)
    async def integrity_error(request: Request, error: Exception) -> JSONResponse:
        logger.info("Database integrity conflict", exc_info=error)
        return _response(
            request,
            status_code=409,
            code="DATABASE_CONFLICT",
            message="请求与现有数据状态冲突",
        )

    @app.exception_handler(OperationalError)
    async def database_unavailable(request: Request, error: Exception) -> JSONResponse:
        logger.exception("Database unavailable")
        return _response(
            request,
            status_code=503,
            code="DATABASE_UNAVAILABLE",
            message="数据库暂时不可用",
        )
