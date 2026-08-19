"""加权初排与修正JND字典序精排的组合测试。"""

from __future__ import annotations

import unittest

from src.profile import (
    PREFERENCE_DIMENSIONS,
    PairwisePreferenceWeightLearner,
    PreferenceDimension,
    PreferencePreset,
    RouteAttributes,
)
from src.recommendation import (
    JndEnhancedRouteRanker,
    JndThresholds,
    RecommendationValidationError,
    RouteConstraints,
)


def _weights(
    *,
    time: float,
    cost: float,
    walking: float = 0.0,
    transfers: float = 0.0,
) -> dict[PreferenceDimension, float]:
    return {
        PreferenceDimension.TIME: time,
        PreferenceDimension.COST: cost,
        PreferenceDimension.WALKING_DISTANCE: walking,
        PreferenceDimension.TRANSFERS: transfers,
    }


def _thresholds(
    *,
    time: float,
    cost: float,
    walking: float = 0.0,
    transfers: float = 0.0,
) -> JndThresholds:
    return JndThresholds(
        time_ratio=time,
        cost_ratio=cost,
        walking_distance_ratio=walking,
        transfers_ratio=transfers,
    )


class JndEnhancedRouteRankerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ranker = JndEnhancedRouteRanker()

    def test_jnd_can_change_weighted_order_inside_shortlist(self) -> None:
        routes = (
            RouteAttributes("fast-expensive", 30, 80, 100, 1),
            RouteAttributes("slow-cheap", 40, 10, 100, 1),
            RouteAttributes("balanced", 33, 50, 100, 1),
        )
        result = self.ranker.rank(
            routes,
            _weights(time=0.6, cost=0.4),
            thresholds=_thresholds(time=0.2, cost=0.1),
            shortlist_size=3,
            top_k=3,
        )

        self.assertEqual(
            [item.route.route_id for item in result.weighted_result.ranked_routes],
            ["slow-cheap", "balanced", "fast-expensive"],
        )
        self.assertEqual(
            [item.route.route_id for item in result.ranked_routes],
            ["balanced", "fast-expensive", "slow-cheap"],
        )
        self.assertEqual(
            result.decisive_dimensions["slow-cheap"],
            PreferenceDimension.TIME,
        )
        self.assertEqual(
            result.decisive_dimensions["balanced"],
            PreferenceDimension.COST,
        )
        self.assertEqual(len(result.comparison_steps), 2)
        time_step, cost_step = result.comparison_steps
        self.assertEqual(time_step.dimension, PreferenceDimension.TIME)
        self.assertEqual(
            time_step.route_ids,
            ("slow-cheap", "balanced", "fast-expensive"),
        )
        self.assertEqual(
            time_step.within_jnd_route_ids,
            ("balanced", "fast-expensive"),
        )
        self.assertEqual(time_step.outside_jnd_route_ids, ("slow-cheap",))
        self.assertEqual(cost_step.dimension, PreferenceDimension.COST)
        self.assertEqual(cost_step.route_ids, ("balanced", "fast-expensive"))

    def test_all_indistinguishable_routes_keep_weighted_order(self) -> None:
        routes = (
            RouteAttributes("fast-expensive", 30, 80, 100, 1),
            RouteAttributes("slow-cheap", 40, 10, 100, 1),
            RouteAttributes("balanced", 33, 50, 100, 1),
        )
        result = self.ranker.rank(
            routes,
            _weights(time=0.6, cost=0.4),
            thresholds=_thresholds(
                time=10.0,
                cost=10.0,
                walking=10.0,
                transfers=10.0,
            ),
            shortlist_size=3,
            top_k=3,
        )

        weighted_ids = [
            item.route.route_id for item in result.weighted_result.ranked_routes
        ]
        self.assertEqual(
            [item.route.route_id for item in result.ranked_routes],
            weighted_ids,
        )
        self.assertTrue(
            all(dimension is None for dimension in result.decisive_dimensions.values())
        )

    def test_difference_equal_to_threshold_is_not_noticeable(self) -> None:
        routes = (
            RouteAttributes("a-time", 10, 20, 100, 0),
            RouteAttributes("b-cost", 12, 10, 100, 0),
        )
        result = self.ranker.rank(
            routes,
            _weights(time=0.9, cost=0.1),
            thresholds=_thresholds(time=0.2, cost=0.0),
            shortlist_size=2,
            top_k=2,
        )

        self.assertEqual(
            [item.route.route_id for item in result.weighted_result.ranked_routes],
            ["a-time", "b-cost"],
        )
        self.assertEqual(
            [item.route.route_id for item in result.ranked_routes],
            ["b-cost", "a-time"],
        )
        self.assertEqual(
            result.decisive_dimensions["a-time"],
            PreferenceDimension.COST,
        )

    def test_adjusted_common_reference_produces_transitive_order(self) -> None:
        # 对应Ceder与Jiang（2020）Table 1/2的三路线反例结构：简单两两
        # 比较可能形成循环，使用全体路线共同最优值后应得到稳定顺序。
        routes = (
            RouteAttributes("A", 11.5, 5, 5, 0),
            RouteAttributes("B", 10, 10, 4, 0),
            RouteAttributes("C", 9, 9, 5, 0),
        )
        result = self.ranker.rank(
            routes,
            _weights(time=0.4, cost=0.3, walking=0.2, transfers=0.1),
            thresholds=_thresholds(
                time=0.2,
                cost=0.2,
                walking=0.1,
                transfers=0.0,
            ),
            shortlist_size=3,
            top_k=3,
        )

        self.assertEqual(
            [item.route.route_id for item in result.ranked_routes],
            ["C", "B", "A"],
        )
        self.assertEqual(result.reference_values[PreferenceDimension.TIME], 9.0)
        self.assertEqual(result.reference_values[PreferenceDimension.COST], 5.0)

    def test_constraints_are_applied_before_jnd_reference_values(self) -> None:
        routes = (
            RouteAttributes("rejected", 1, 200, 0, 0),
            RouteAttributes("feasible-a", 30, 20, 100, 0),
            RouteAttributes("feasible-b", 35, 15, 200, 1),
        )
        result = self.ranker.rank(
            routes,
            _weights(time=0.7, cost=0.3),
            thresholds=_thresholds(time=0.1, cost=0.1),
            constraints=RouteConstraints(max_total_cost=100),
            shortlist_size=3,
            top_k=2,
        )

        self.assertEqual(result.reference_values[PreferenceDimension.TIME], 30.0)
        self.assertEqual(
            [item.route.route_id for item in result.rejected_routes],
            ["rejected"],
        )
        self.assertEqual(result.shortlist_size, 2)

    def test_only_weighted_top_n_routes_enter_jnd_reranking(self) -> None:
        routes = (
            RouteAttributes("one", 10, 10, 100, 0),
            RouteAttributes("two", 20, 20, 100, 0),
            RouteAttributes("three", 30, 30, 100, 0),
            RouteAttributes("outside", 40, 40, 100, 0),
        )
        result = self.ranker.rank(
            routes,
            _weights(time=0.5, cost=0.5),
            thresholds=_thresholds(time=0.1, cost=0.1),
            shortlist_size=3,
            top_k=2,
        )

        self.assertEqual(result.feasible_count, 4)
        self.assertEqual(result.shortlist_size, 3)
        self.assertNotIn("outside", result.decisive_dimensions)
        self.assertNotIn(
            "outside",
            [item.route.route_id for item in result.ranked_routes],
        )

    def test_no_feasible_route_returns_empty_jnd_result(self) -> None:
        result = self.ranker.rank(
            (RouteAttributes("too-expensive", 20, 100, 100, 0),),
            _weights(time=0.5, cost=0.5),
            thresholds=_thresholds(time=0.1, cost=0.1),
            constraints=RouteConstraints(max_total_cost=10),
            shortlist_size=3,
            top_k=2,
        )

        self.assertIsNone(result.recommended)
        self.assertEqual(result.shortlist_size, 0)
        self.assertEqual(dict(result.reference_values), {})
        self.assertEqual(len(result.rejected_routes), 1)

    def test_equal_weights_use_stable_dimension_order(self) -> None:
        result = self.ranker.rank(
            (RouteAttributes("only", 20, 10, 100, 0),),
            {dimension: 1.0 for dimension in PREFERENCE_DIMENSIONS},
            thresholds=_thresholds(
                time=0.1,
                cost=0.1,
                walking=0.1,
                transfers=0.1,
            ),
            shortlist_size=1,
            top_k=1,
        )

        self.assertEqual(result.attribute_priority, PREFERENCE_DIMENSIONS)

    def test_learned_profile_can_be_used_without_converting_weights(self) -> None:
        profile = PairwisePreferenceWeightLearner().fit(
            (),
            preference_preset=PreferencePreset.TIME_PRIORITY,
        )
        result = self.ranker.rank(
            (
                RouteAttributes("fast", 20, 30, 100, 0),
                RouteAttributes("cheap", 30, 10, 100, 0),
            ),
            profile,
            thresholds=_thresholds(time=0.1, cost=0.1),
            shortlist_size=2,
            top_k=1,
        )

        self.assertEqual(result.attribute_priority[0], PreferenceDimension.TIME)
        self.assertEqual(len(result.ranked_routes), 1)

    def test_invalid_thresholds_and_sizes_are_rejected(self) -> None:
        with self.assertRaises(RecommendationValidationError):
            _thresholds(time=-0.1, cost=0.1)
        with self.assertRaises(RecommendationValidationError):
            self.ranker.rank(
                (RouteAttributes("only", 20, 10, 100, 0),),
                _weights(time=0.5, cost=0.5),
                thresholds=_thresholds(time=0.1, cost=0.1),
                shortlist_size=2,
                top_k=3,
            )


if __name__ == "__main__":
    unittest.main()
