"""应用服务的输入命令，不依赖 HTTP 或数据库框架。"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class RouteInput:
    route_id: str
    total_time_minutes: float
    total_cost: float
    model_cost_unit: str
    walking_distance_meters: float
    transfer_count: int


@dataclass(frozen=True, slots=True)
class ConstraintsInput:
    max_total_time_minutes: float | None = None
    max_total_cost: float | None = None
    max_walking_distance_meters: float | None = None
    max_transfer_count: int | None = None


@dataclass(frozen=True, slots=True)
class JndInput:
    shortlist_size: int
    top_k: int
    time_ratio: float
    cost_ratio: float
    walking_distance_ratio: float
    transfers_ratio: float


@dataclass(frozen=True, slots=True)
class RecommendationCommand:
    user_id: UUID
    ranking_mode: str
    routes: tuple[RouteInput, ...]
    constraints: ConstraintsInput
    top_k: int | None = None
    jnd: JndInput | None = None


@dataclass(frozen=True, slots=True)
class ChoiceCommand:
    user_id: UUID
    chosen_route: RouteInput
    rejected_route: RouteInput
