"""基于论文加权多指标方法的候选路线个性化排序。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite

from ..profile.models import (
    PREFERENCE_DIMENSIONS,
    PreferenceDimension,
    PreferenceLearningResult,
    RouteAttributes,
)
from ..profile.normalization import NORMALIZATION_SCALES
from .exceptions import RecommendationValidationError
from .models import (
    RankedRoute,
    RejectedRoute,
    RouteConstraints,
    RouteRankingResult,
)


PreferenceWeights = Mapping[PreferenceDimension, float] | PreferenceLearningResult


class PersonalizedRouteRanker:
    """过滤候选路线，并按个人四维画像返回Top-K。

    排序采用 Preference-Aware Multimodal Journey Planner（2026）公式（1）至
    （4）的加权和模型；候选路线硬约束采用 Jiang 与 Ceder（2019）公式（3）。
    四项路线特征复用FAVOUR学习阶段的固定归一化尺度，以避免训练与推荐阶段
    的特征语义不一致。
    """

    @staticmethod
    def _resolve_weights(preference: PreferenceWeights) -> dict[PreferenceDimension, float]:
        raw_weights = preference.weights if isinstance(
            preference, PreferenceLearningResult
        ) else preference
        if set(raw_weights) != set(PREFERENCE_DIMENSIONS):
            raise RecommendationValidationError("偏好权重必须包含完整的四个维度")

        weights = {
            dimension: float(raw_weights[dimension])
            for dimension in PREFERENCE_DIMENSIONS
        }
        if any(not isfinite(value) or value < 0.0 for value in weights.values()):
            raise RecommendationValidationError("偏好权重必须是有限的非负数")
        total = sum(weights.values())
        if total <= 0.0:
            raise RecommendationValidationError("偏好权重之和必须大于0")
        return {dimension: weights[dimension] / total for dimension in PREFERENCE_DIMENSIONS}

    @staticmethod
    def _normalized_attributes(
        route: RouteAttributes,
    ) -> dict[PreferenceDimension, float]:
        return {
            dimension: route.value_for(dimension) / NORMALIZATION_SCALES[dimension]
            for dimension in PREFERENCE_DIMENSIONS
        }

    @staticmethod
    def _validate_top_k(top_k: int | None) -> None:
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0
        ):
            raise RecommendationValidationError("top_k必须是正整数或None")

    def rank(
        self,
        routes: Sequence[RouteAttributes],
        preference: PreferenceWeights,
        *,
        constraints: RouteConstraints | None = None,
        top_k: int | None = None,
    ) -> RouteRankingResult:
        """返回满足硬约束的候选路线及其个性化加权代价。

        ``routes`` 应由上游多模式路径服务生成；本方法只负责论文所述的候选
        过滤和个性化排序，不在交通网络中重新执行路径搜索。
        """

        candidates = tuple(routes)
        if not candidates:
            raise RecommendationValidationError("候选路线不能为空")
        route_ids = [route.route_id for route in candidates]
        if len(set(route_ids)) != len(route_ids):
            raise RecommendationValidationError("候选路线的route_id不能重复")
        self._validate_top_k(top_k)
        weights = self._resolve_weights(preference)
        active_constraints = constraints or RouteConstraints()

        rejected: list[RejectedRoute] = []
        scored: list[
            tuple[
                RouteAttributes,
                float,
                dict[PreferenceDimension, float],
                dict[PreferenceDimension, float],
            ]
        ] = []
        for route in candidates:
            violations = active_constraints.violations(route)
            if violations:
                rejected.append(RejectedRoute(route, violations))
                continue

            normalized = self._normalized_attributes(route)
            contributions = {
                dimension: weights[dimension] * normalized[dimension]
                for dimension in PREFERENCE_DIMENSIONS
            }
            scored.append((route, sum(contributions.values()), normalized, contributions))

        scored.sort(key=lambda item: (item[1], item[0].route_id))
        feasible_count = len(scored)
        if scored:
            averages = {
                dimension: sum(item[2][dimension] for item in scored) / len(scored)
                for dimension in PREFERENCE_DIMENSIONS
            }
        else:
            averages = {dimension: 0.0 for dimension in PREFERENCE_DIMENSIONS}

        if top_k is not None:
            scored = scored[:top_k]

        ranked = []
        for rank, (route, cost, normalized, contributions) in enumerate(scored, start=1):
            advantages = sorted(
                (
                    (
                        weights[dimension]
                        * (averages[dimension] - normalized[dimension]),
                        dimension,
                    )
                    for dimension in PREFERENCE_DIMENSIONS
                ),
                key=lambda item: (-item[0], item[1].value),
            )
            advantage_dimensions = tuple(
                dimension for advantage, dimension in advantages if advantage > 1e-12
            )
            ranked.append(
                RankedRoute(
                    rank=rank,
                    route=route,
                    personalized_cost=cost,
                    normalized_attributes=normalized,
                    weighted_contributions=contributions,
                    advantage_dimensions=advantage_dimensions,
                )
            )

        return RouteRankingResult(
            ranked_routes=tuple(ranked),
            rejected_routes=tuple(rejected),
            normalized_weights=weights,
            candidate_count=len(candidates),
            feasible_count=feasible_count,
        )
