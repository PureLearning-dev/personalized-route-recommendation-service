"""加权初排与修正JND字典序精排的组合测试。"""

from __future__ import annotations

import unittest
from collections.abc import Sequence
from itertools import permutations

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
    RankedRoute,
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


def _route_ids(ranked_routes: Sequence[RankedRoute]) -> list[str]:
    return [item.route.route_id for item in ranked_routes]


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
            _route_ids(result.weighted_result.ranked_routes),
            ["slow-cheap", "balanced", "fast-expensive"],
        )
        self.assertEqual(
            _route_ids(result.ranked_routes),
            ["balanced", "fast-expensive", "slow-cheap"],
        )
        self.assertEqual([item.rank for item in result.ranked_routes], [1, 2, 3])
        self.assertEqual(
            result.attribute_priority,
            (
                PreferenceDimension.TIME,
                PreferenceDimension.COST,
                PreferenceDimension.WALKING_DISTANCE,
                PreferenceDimension.TRANSFERS,
            ),
        )
        self.assertEqual(
            dict(result.reference_values),
            {
                PreferenceDimension.TIME: 30.0,
                PreferenceDimension.COST: 10.0,
                PreferenceDimension.WALKING_DISTANCE: 100.0,
                PreferenceDimension.TRANSFERS: 1.0,
            },
        )
        self.assertEqual(
            dict(result.decisive_dimensions),
            {
                "slow-cheap": PreferenceDimension.TIME,
                "balanced": PreferenceDimension.COST,
                "fast-expensive": PreferenceDimension.COST,
            },
        )

        self.assertEqual(len(result.comparison_steps), 2)
        time_step, cost_step = result.comparison_steps
        self.assertEqual(
            (
                time_step.priority_level,
                time_step.dimension,
                time_step.reference_value,
                time_step.threshold_ratio,
                time_step.noticeable_difference,
            ),
            (1, PreferenceDimension.TIME, 30.0, 0.2, 6.0),
        )
        self.assertEqual(
            time_step.route_ids,
            ("slow-cheap", "balanced", "fast-expensive"),
        )
        self.assertEqual(
            time_step.within_jnd_route_ids,
            ("balanced", "fast-expensive"),
        )
        self.assertEqual(time_step.outside_jnd_route_ids, ("slow-cheap",))
        self.assertEqual(
            (
                cost_step.priority_level,
                cost_step.dimension,
                cost_step.reference_value,
                cost_step.threshold_ratio,
                cost_step.noticeable_difference,
            ),
            (2, PreferenceDimension.COST, 10.0, 0.1, 1.0),
        )
        self.assertEqual(cost_step.route_ids, ("balanced", "fast-expensive"))
        self.assertEqual(cost_step.within_jnd_route_ids, ())
        self.assertEqual(
            cost_step.outside_jnd_route_ids,
            ("balanced", "fast-expensive"),
        )

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

        weighted_ids = _route_ids(result.weighted_result.ranked_routes)
        self.assertEqual(_route_ids(result.ranked_routes), weighted_ids)
        self.assertTrue(
            all(dimension is None for dimension in result.decisive_dimensions.values())
        )
        self.assertEqual(
            tuple(step.dimension for step in result.comparison_steps),
            result.attribute_priority,
        )
        self.assertTrue(
            all(not step.outside_jnd_route_ids for step in result.comparison_steps)
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
            _route_ids(result.weighted_result.ranked_routes),
            ["a-time", "b-cost"],
        )
        self.assertEqual(_route_ids(result.ranked_routes), ["b-cost", "a-time"])
        time_step, cost_step = result.comparison_steps
        self.assertEqual(time_step.noticeable_difference, 2.0)
        self.assertEqual(time_step.within_jnd_route_ids, ("a-time", "b-cost"))
        self.assertEqual(time_step.outside_jnd_route_ids, ())
        self.assertEqual(cost_step.within_jnd_route_ids, ("b-cost",))
        self.assertEqual(cost_step.outside_jnd_route_ids, ("a-time",))
        self.assertEqual(
            result.decisive_dimensions["a-time"],
            PreferenceDimension.COST,
        )

    def test_zero_reference_makes_any_positive_difference_noticeable(self) -> None:
        routes = (
            RouteAttributes("zero-transfer", 20, 10, 100, 0),
            RouteAttributes("one-transfer", 10, 10, 100, 1),
        )
        result = self.ranker.rank(
            routes,
            _weights(time=0.1, cost=0.0, transfers=0.9),
            thresholds=_thresholds(
                time=10.0,
                cost=10.0,
                walking=10.0,
                transfers=0.5,
            ),
            shortlist_size=2,
            top_k=2,
        )

        transfer_step = result.comparison_steps[0]
        self.assertEqual(transfer_step.dimension, PreferenceDimension.TRANSFERS)
        self.assertEqual(transfer_step.reference_value, 0.0)
        self.assertEqual(transfer_step.noticeable_difference, 0.0)
        self.assertEqual(transfer_step.within_jnd_route_ids, ("zero-transfer",))
        self.assertEqual(transfer_step.outside_jnd_route_ids, ("one-transfer",))

    def test_equal_noticeable_values_continue_with_next_dimension(self) -> None:
        routes = (
            RouteAttributes("time-best", 10, 100, 100, 0),
            RouteAttributes("tie-expensive", 20, 20, 100, 0),
            RouteAttributes("tie-cheap", 20, 10, 100, 0),
        )
        result = self.ranker.rank(
            routes,
            _weights(time=0.6, cost=0.4),
            thresholds=_thresholds(time=0.1, cost=0.0),
            shortlist_size=3,
            top_k=3,
        )

        self.assertEqual(
            _route_ids(result.weighted_result.ranked_routes),
            ["tie-cheap", "tie-expensive", "time-best"],
        )
        self.assertEqual(
            _route_ids(result.ranked_routes),
            ["time-best", "tie-cheap", "tie-expensive"],
        )
        time_step, cost_step = result.comparison_steps
        self.assertEqual(
            time_step.outside_jnd_route_ids,
            ("tie-cheap", "tie-expensive"),
        )
        self.assertEqual(
            cost_step.route_ids,
            ("tie-cheap", "tie-expensive"),
        )
        self.assertEqual(cost_step.within_jnd_route_ids, ("tie-cheap",))
        self.assertEqual(cost_step.outside_jnd_route_ids, ("tie-expensive",))

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

        self.assertEqual(_route_ids(result.ranked_routes), ["C", "B", "A"])
        self.assertEqual(
            dict(result.reference_values),
            {
                PreferenceDimension.TIME: 9.0,
                PreferenceDimension.COST: 5.0,
                PreferenceDimension.WALKING_DISTANCE: 4.0,
                PreferenceDimension.TRANSFERS: 0.0,
            },
        )
        time_step, cost_step = result.comparison_steps
        self.assertEqual(time_step.within_jnd_route_ids, ("C", "B"))
        self.assertEqual(time_step.outside_jnd_route_ids, ("A",))
        self.assertEqual(cost_step.outside_jnd_route_ids, ("C", "B"))

        # 非传递的两两比较器可能随输入排列产生不同结果；共同参考值算法应对
        # 同一候选集合始终给出同一全序。
        for route_order in permutations(routes):
            with self.subTest(
                route_order=tuple(route.route_id for route in route_order)
            ):
                permuted_result = self.ranker.rank(
                    route_order,
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
                    _route_ids(permuted_result.ranked_routes),
                    ["C", "B", "A"],
                )

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
        self.assertTrue(
            all(
                "rejected" not in step.route_ids
                for step in result.comparison_steps
            )
        )

    def test_only_weighted_top_n_routes_enter_jnd_reranking(self) -> None:
        routes = (
            RouteAttributes("one", 10, 10, 100, 0),
            RouteAttributes("two", 20, 20, 100, 0),
            RouteAttributes("three", 30, 30, 100, 0),
            # 时间最优但费用极高，因此加权初排落在Top-N之外。若共同参考值
            # 错误地从全部可行路线计算，时间参考值会从10变成1。
            RouteAttributes("outside", 1, 100, 100, 0),
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
        self.assertEqual(result.reference_values[PreferenceDimension.TIME], 10.0)
        self.assertNotIn("outside", result.decisive_dimensions)
        self.assertNotIn(
            "outside",
            _route_ids(result.ranked_routes),
        )
        self.assertTrue(
            all("outside" not in step.route_ids for step in result.comparison_steps)
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
        self.assertEqual(dict(result.decisive_dimensions), {})
        self.assertEqual(result.comparison_steps, ())
        self.assertEqual(result.feasible_count, 0)
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
        for invalid_threshold in (-0.1, float("nan"), float("inf")):
            with self.subTest(threshold=invalid_threshold):
                with self.assertRaises(RecommendationValidationError):
                    _thresholds(time=invalid_threshold, cost=0.1)

        route = (RouteAttributes("only", 20, 10, 100, 0),)
        valid_thresholds = _thresholds(time=0.1, cost=0.1)
        for field_name, invalid_value in (
            ("shortlist_size", 0),
            ("shortlist_size", -1),
            ("shortlist_size", 1.5),
            ("shortlist_size", True),
            ("top_k", 0),
            ("top_k", -1),
            ("top_k", 1.5),
            ("top_k", False),
        ):
            sizes = {"shortlist_size": 2, "top_k": 1}
            sizes[field_name] = invalid_value
            with self.subTest(field=field_name, value=invalid_value):
                with self.assertRaisesRegex(
                    RecommendationValidationError,
                    field_name,
                ):
                    self.ranker.rank(
                        route,
                        _weights(time=0.5, cost=0.5),
                        thresholds=valid_thresholds,
                        **sizes,  # type: ignore[arg-type]
                    )

        with self.assertRaisesRegex(
            RecommendationValidationError,
            "top_k不能大于shortlist_size",
        ):
            self.ranker.rank(
                route,
                _weights(time=0.5, cost=0.5),
                thresholds=valid_thresholds,
                shortlist_size=2,
                top_k=3,
            )


if __name__ == "__main__":
    unittest.main()
