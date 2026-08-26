"""FastAPI 应用工厂。"""

from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import FastAPI, Request

from ..config import get_settings
from .errors import install_error_handlers
from .routers import health, recommendations, users


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    app = FastAPI(
        title="Personalized Route Recommendation API",
        version="1.0.0",
        description="保存用户画像并对候选多模式路线进行个性化排序。",
        swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID", str(uuid4()))
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    install_error_handlers(app)
    app.include_router(health.router)
    app.include_router(users.router, prefix="/v1")
    app.include_router(recommendations.router, prefix="/v1")
    return app


app = create_app()
