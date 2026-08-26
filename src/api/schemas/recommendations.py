from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import Field, model_validator

from .common import ApiModel
from .profiles import ProfileResponse


class RouteInputSchema(ApiModel):
    route_id: str = Field(min_length=1, max_length=200)
    total_time_minutes: float = Field(ge=0)
    total_cost: float = Field(ge=0)
    model_cost_unit: str = Field(min_length=1, max_length=16)
    walking_distance_meters: float = Field(ge=0)
    transfer_count: int = Field(ge=0)


class ConstraintsSchema(ApiModel):
    max_total_time_minutes: float | None = Field(default=None, ge=0)
    max_total_cost: float | None = Field(default=None, ge=0)
    max_walking_distance_meters: float | None = Field(default=None, ge=0)
    max_transfer_count: int | None = Field(default=None, ge=0)


class JndThresholdSchema(ApiModel):
    time_ratio: float = Field(ge=0)
    cost_ratio: float = Field(ge=0)
    walking_distance_ratio: float = Field(ge=0)
    transfers_ratio: float = Field(ge=0)


class JndSchema(ApiModel):
    shortlist_size: int = Field(gt=0)
    top_k: int = Field(gt=0)
    thresholds: JndThresholdSchema

    @model_validator(mode="after")
    def validate_sizes(self) -> JndSchema:
        if self.top_k > self.shortlist_size:
            raise ValueError("top_k 不能大于 shortlist_size")
        return self


class RecommendationCreate(ApiModel):
    user_id: UUID
    ranking_mode: Literal["weighted", "jnd"]
    routes: list[RouteInputSchema] = Field(min_length=1)
    constraints: ConstraintsSchema = Field(default_factory=ConstraintsSchema)
    top_k: int | None = Field(default=None, gt=0)
    jnd: JndSchema | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> RecommendationCreate:
        if self.ranking_mode == "weighted" and self.jnd is not None:
            raise ValueError("weighted 模式不能携带 jnd 参数")
        if self.ranking_mode == "jnd":
            if self.jnd is None:
                raise ValueError("jnd 模式必须携带 jnd 参数")
            if self.top_k is not None:
                raise ValueError("jnd 模式的 top_k 必须写在 jnd 对象中")
        return self


class RankedRouteResponse(ApiModel):
    rank: int
    route: dict[str, Any]
    personalized_cost: float
    normalized_attributes: dict[str, float]
    weighted_contributions: dict[str, float]
    advantage_dimensions: list[str]


class RejectedRouteResponse(ApiModel):
    route_id: str
    violated_dimensions: list[str]


class RecommendationResponse(ApiModel):
    profile: ProfileResponse
    ranking_mode: str
    candidate_count: int
    feasible_count: int
    ranked_routes: list[RankedRouteResponse]
    rejected_routes: list[RejectedRouteResponse]
    explanation: dict[str, Any]
