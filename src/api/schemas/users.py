from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from .common import ApiModel
from .profiles import ProfileResponse


class InitialProfileInput(ApiModel):
    mode: Literal["default", "preset"]
    preset: (
        Literal[
            "balanced",
            "time_priority",
            "cost_priority",
            "low_walking",
            "low_transfers",
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def validate_shape(self) -> InitialProfileInput:
        if self.mode == "default" and self.preset is not None:
            raise ValueError("default 模式不能携带 preset")
        if self.mode == "preset" and self.preset is None:
            raise ValueError("preset 模式必须指定 preset")
        return self


class UserCreate(ApiModel):
    external_user_id: str = Field(min_length=1, max_length=200)
    initial_profile: InitialProfileInput


class UserResponse(ApiModel):
    id: UUID
    external_user_id: str
    initialization_mode: str
    preset_name: str | None
    created_at: datetime
    profile: ProfileResponse
