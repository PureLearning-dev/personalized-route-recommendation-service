"""候选路线过滤与个性化排序使用的数据模型。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

from ..profile.models import PreferenceDimension, RouteAttributes
from .exceptions import RecommendationValidationError


def _optional_nonnegative_float(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    resolved = float(value)
    if not isfinite(resolved) or resolved < 0.0:
        raise RecommendationValidationError(f"{field_name}必须是有限的非负数或None")
    return resolved


def _nonnegative_float(value: float, field_name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved < 0.0:
        raise RecommendationValidationError(f"{field_name}必须是有限的非负数")
    return resolved


@dataclass(frozen=True, slots=True)
class RouteConstraints:
    """用户本次出行对四项代价属性设置的最大可接受值。

    对应 Jiang 与 Ceder（2019）公式（3）的个人最大/最小可接受值筛选。
    当前四个指标均为代价，所以这里只需要最大值约束。
    """

    max_total_time_minutes: float | None = None
    max_total_cost: float | None = None
    max_walking_distance_meters: float | None = None
    max_transfer_count: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "max_total_time_minutes",
            "max_total_cost",
            "max_walking_distance_meters",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_nonnegative_float(getattr(self, field_name), field_name),
            )

        if self.max_transfer_count is not None and (
            isinstance(self.max_transfer_count, bool)
            or not isinstance(self.max_transfer_count, int)
            or self.max_transfer_count < 0
        ):
            raise RecommendationValidationError(
                "max_transfer_count必须是非负整数或None"
            )

    def violations(self, route: RouteAttributes) -> tuple[PreferenceDimension, ...]:
        """返回路线超过最大可接受值的维度。"""

        violated: list[PreferenceDimension] = []
        if (
            self.max_total_time_minutes is not None
            and route.total_time_minutes > self.max_total_time_minutes
        ):
            violated.append(PreferenceDimension.TIME)
        if self.max_total_cost is not None and route.total_cost > self.max_total_cost:
            violated.append(PreferenceDimension.COST)
        if (
            self.max_walking_distance_meters is not None
            and route.walking_distance_meters > self.max_walking_distance_meters
        ):
            violated.append(PreferenceDimension.WALKING_DISTANCE)
        if (
            self.max_transfer_count is not None
            and route.transfer_count > self.max_transfer_count
        ):
            violated.append(PreferenceDimension.TRANSFERS)
        return tuple(violated)


@dataclass(frozen=True, slots=True)
class RejectedRoute:
    """因违反本次出行硬约束而被排除的候选路线。"""

    route: RouteAttributes
    violated_dimensions: tuple[PreferenceDimension, ...]

    def __post_init__(self) -> None:
        if not self.violated_dimensions:
            raise RecommendationValidationError("被排除的路线必须至少违反一个约束")


@dataclass(frozen=True, slots=True)
class RankedRoute:
    """一条候选路线的个性化排序结果。

    ``personalized_cost`` 越小越符合当前画像。``weighted_contributions`` 保存
    四项归一化代价乘以画像权重后的分项值，其和等于个性化代价。
    """

    rank: int
    route: RouteAttributes
    personalized_cost: float
    normalized_attributes: Mapping[PreferenceDimension, float]
    weighted_contributions: Mapping[PreferenceDimension, float]
    advantage_dimensions: tuple[PreferenceDimension, ...]

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise RecommendationValidationError("路线名次必须是正整数")
        if not isfinite(self.personalized_cost) or self.personalized_cost < 0.0:
            raise RecommendationValidationError("个性化代价必须是有限的非负数")
        object.__setattr__(
            self,
            "normalized_attributes",
            MappingProxyType(dict(self.normalized_attributes)),
        )
        object.__setattr__(
            self,
            "weighted_contributions",
            MappingProxyType(dict(self.weighted_contributions)),
        )


@dataclass(frozen=True, slots=True)
class RouteRankingResult:
    """一次候选路线过滤与Top-K个性化排序的完整结果。"""

    ranked_routes: tuple[RankedRoute, ...]
    rejected_routes: tuple[RejectedRoute, ...]
    normalized_weights: Mapping[PreferenceDimension, float]
    candidate_count: int
    feasible_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "normalized_weights",
            MappingProxyType(dict(self.normalized_weights)),
        )

    @property
    def recommended(self) -> RankedRoute | None:
        """返回当前最推荐的路线；无可行路线时返回None。"""

        return self.ranked_routes[0] if self.ranked_routes else None


@dataclass(frozen=True, slots=True)
class JndThresholds:
    """四项路线属性的Weber JND比例阈值。

    例如 ``time_ratio=0.1`` 表示相对当前候选集中最短时间增加超过10%时，
    才认为时间差异可以被感知。阈值是业务配置，不在这里假设统一默认值。
    """

    time_ratio: float
    cost_ratio: float
    walking_distance_ratio: float
    transfers_ratio: float

    def __post_init__(self) -> None:
        for field_name in (
            "time_ratio",
            "cost_ratio",
            "walking_distance_ratio",
            "transfers_ratio",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_float(getattr(self, field_name), field_name),
            )

    def ratio_for(self, dimension: PreferenceDimension) -> float:
        return {
            PreferenceDimension.TIME: self.time_ratio,
            PreferenceDimension.COST: self.cost_ratio,
            PreferenceDimension.WALKING_DISTANCE: self.walking_distance_ratio,
            PreferenceDimension.TRANSFERS: self.transfers_ratio,
        }[dimension]


@dataclass(frozen=True, slots=True)
class JndComparisonStep:
    """一次修正JND字典序比较的过程记录。"""

    priority_level: int
    dimension: PreferenceDimension
    route_ids: tuple[str, ...]
    reference_value: float
    threshold_ratio: float
    noticeable_difference: float
    within_jnd_route_ids: tuple[str, ...]
    outside_jnd_route_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.priority_level <= 0:
            raise RecommendationValidationError("JND比较层级必须是正整数")
        if not self.route_ids:
            raise RecommendationValidationError("JND比较过程必须至少包含一条路线")
        partition = self.within_jnd_route_ids + self.outside_jnd_route_ids
        if len(partition) != len(self.route_ids) or set(partition) != set(self.route_ids):
            raise RecommendationValidationError("JND范围内外路线必须完整覆盖当前比较组")


@dataclass(frozen=True, slots=True)
class JndEnhancedRankingResult:
    """加权初排后，对前N条路线进行JND精排的结果。

    ``decisive_dimensions`` 记录每条精排候选路线第一次落在JND最优范围之外
    的指标；值为 ``None`` 表示它在实际比较过程中始终留在JND最优组。
    """

    ranked_routes: tuple[RankedRoute, ...]
    weighted_result: RouteRankingResult
    attribute_priority: tuple[PreferenceDimension, ...]
    thresholds: JndThresholds
    reference_values: Mapping[PreferenceDimension, float]
    decisive_dimensions: Mapping[str, PreferenceDimension | None]
    comparison_steps: tuple[JndComparisonStep, ...]
    shortlist_size: int

    def __post_init__(self) -> None:
        if self.shortlist_size < 0:
            raise RecommendationValidationError("JND精排路线数量不能为负数")
        if len(self.ranked_routes) > self.shortlist_size:
            raise RecommendationValidationError("最终路线数量不能超过JND精排路线数量")
        if (
            len(self.attribute_priority) != len(PreferenceDimension)
            or set(self.attribute_priority) != set(PreferenceDimension)
        ):
            raise RecommendationValidationError("JND指标优先级必须包含完整的四个维度")
        object.__setattr__(
            self,
            "reference_values",
            MappingProxyType(dict(self.reference_values)),
        )
        object.__setattr__(
            self,
            "decisive_dimensions",
            MappingProxyType(dict(self.decisive_dimensions)),
        )

    @property
    def recommended(self) -> RankedRoute | None:
        return self.ranked_routes[0] if self.ranked_routes else None

    @property
    def rejected_routes(self) -> tuple[RejectedRoute, ...]:
        return self.weighted_result.rejected_routes

    @property
    def candidate_count(self) -> int:
        return self.weighted_result.candidate_count

    @property
    def feasible_count(self) -> int:
        return self.weighted_result.feasible_count

    @property
    def normalized_weights(self) -> Mapping[PreferenceDimension, float]:
        return self.weighted_result.normalized_weights
