"""在加权初排结果上执行修正后的JND字典序精排。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import groupby

from ..profile.models import PREFERENCE_DIMENSIONS, PreferenceDimension, RouteAttributes
from .exceptions import RecommendationValidationError
from .models import (
    JndComparisonStep,
    JndEnhancedRankingResult,
    JndThresholds,
    RankedRoute,
    RouteConstraints,
)
from .ranking import PersonalizedRouteRanker, PreferenceWeights


class JndEnhancedRouteRanker:
    """执行“硬约束过滤—加权初排—前N条JND精排—Top-K”。

    JND部分参考 Ceder 与 Jiang（2020）修正后的比较方法：每个指标都以
    精排候选集中的最优值为共同参照，避免简单两两比较造成不传递。画像权重
    只用于确定指标优先级；JND比较使用路线的原始属性值。
    """

    def __init__(self, weighted_ranker: PersonalizedRouteRanker | None = None) -> None:
        self._weighted_ranker = weighted_ranker or PersonalizedRouteRanker()

    @staticmethod
    def _validate_sizes(shortlist_size: int, top_k: int) -> None:
        for value, field_name in (
            (shortlist_size, "shortlist_size"),
            (top_k, "top_k"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RecommendationValidationError(f"{field_name}必须是正整数")
        if top_k > shortlist_size:
            raise RecommendationValidationError("top_k不能大于shortlist_size")

    @staticmethod
    def _attribute_priority(
        weights: Mapping[PreferenceDimension, float],
    ) -> tuple[PreferenceDimension, ...]:
        original_order = {
            dimension: index
            for index, dimension in enumerate(PREFERENCE_DIMENSIONS)
        }
        return tuple(
            sorted(
                PREFERENCE_DIMENSIONS,
                key=lambda dimension: (
                    -weights[dimension],
                    original_order[dimension],
                ),
            )
        )

    @staticmethod
    def _reference_values(
        shortlist: Sequence[RankedRoute],
    ) -> dict[PreferenceDimension, float]:
        return {
            dimension: min(item.route.value_for(dimension) for item in shortlist)
            for dimension in PREFERENCE_DIMENSIONS
        }

    @classmethod
    def _rerank_group(
        cls,
        routes: Sequence[RankedRoute],
        priority: tuple[PreferenceDimension, ...],
        priority_index: int,
        reference_values: Mapping[PreferenceDimension, float],
        thresholds: JndThresholds,
        decisive_dimensions: dict[str, PreferenceDimension | None],
        comparison_steps: list[JndComparisonStep],
    ) -> list[RankedRoute]:
        """递归实现Algorithm 1，并以加权顺序作为不可区分时的稳定后备。"""

        if len(routes) <= 1 or priority_index >= len(priority):
            return list(routes)

        dimension = priority[priority_index]
        reference = reference_values[dimension]
        noticeable_limit = thresholds.ratio_for(dimension) * reference
        indistinguishable: list[RankedRoute] = []
        noticeable: list[RankedRoute] = []

        for item in routes:
            difference = item.route.value_for(dimension) - reference
            if difference > noticeable_limit:
                noticeable.append(item)
                if decisive_dimensions[item.route.route_id] is None:
                    decisive_dimensions[item.route.route_id] = dimension
            else:
                indistinguishable.append(item)

        comparison_steps.append(
            JndComparisonStep(
                priority_level=priority_index + 1,
                dimension=dimension,
                route_ids=tuple(item.route.route_id for item in routes),
                reference_value=reference,
                threshold_ratio=thresholds.ratio_for(dimension),
                noticeable_difference=noticeable_limit,
                within_jnd_route_ids=tuple(
                    item.route.route_id for item in indistinguishable
                ),
                outside_jnd_route_ids=tuple(item.route.route_id for item in noticeable),
            )
        )

        ordered = cls._rerank_group(
            indistinguishable,
            priority,
            priority_index + 1,
            reference_values,
            thresholds,
            decisive_dimensions,
            comparison_steps,
        )

        # 当前指标已能区分的路线按该指标升序排列；属性值完全相同时，才继续
        # 比较下一个指标。输入本身保持加权顺序，因此最终后备结果稳定可复现。
        noticeable.sort(key=lambda item: item.route.value_for(dimension))
        for _, equal_value_group in groupby(
            noticeable,
            key=lambda item: item.route.value_for(dimension),
        ):
            ordered.extend(
                cls._rerank_group(
                    tuple(equal_value_group),
                    priority,
                    priority_index + 1,
                    reference_values,
                    thresholds,
                    decisive_dimensions,
                    comparison_steps,
                )
            )
        return ordered

    def rank(
        self,
        routes: Sequence[RouteAttributes],
        preference: PreferenceWeights,
        *,
        thresholds: JndThresholds,
        shortlist_size: int,
        top_k: int,
        constraints: RouteConstraints | None = None,
    ) -> JndEnhancedRankingResult:
        """返回加权初排与JND精排组合后的Top-K路线。"""

        self._validate_sizes(shortlist_size, top_k)
        weighted_result = self._weighted_ranker.rank(
            routes,
            preference,
            constraints=constraints,
            top_k=None,
        )
        priority = self._attribute_priority(weighted_result.normalized_weights)
        shortlist = weighted_result.ranked_routes[:shortlist_size]

        if not shortlist:
            return JndEnhancedRankingResult(
                ranked_routes=(),
                weighted_result=weighted_result,
                attribute_priority=priority,
                thresholds=thresholds,
                reference_values={},
                decisive_dimensions={},
                comparison_steps=(),
                shortlist_size=0,
            )

        reference_values = self._reference_values(shortlist)
        decisive_dimensions: dict[str, PreferenceDimension | None] = {
            item.route.route_id: None for item in shortlist
        }
        comparison_steps: list[JndComparisonStep] = []
        reranked = self._rerank_group(
            shortlist,
            priority,
            0,
            reference_values,
            thresholds,
            decisive_dimensions,
            comparison_steps,
        )
        final_routes = tuple(
            replace(item, rank=rank)
            for rank, item in enumerate(reranked[:top_k], start=1)
        )
        return JndEnhancedRankingResult(
            ranked_routes=final_routes,
            weighted_result=weighted_result,
            attribute_priority=priority,
            thresholds=thresholds,
            reference_values=reference_values,
            decisive_dimensions=decisive_dimensions,
            comparison_steps=tuple(comparison_steps),
            shortlist_size=len(shortlist),
        )
