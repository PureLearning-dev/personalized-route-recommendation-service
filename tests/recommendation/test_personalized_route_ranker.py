"""论文加权排序、硬约束过滤及FAVOUR集成测试。"""

from __future__ import annotations

import unittest

from src.profile import (
    PREFERENCE_DIMENSIONS,
    PairwisePreference,
    PairwisePreferenceWeightLearner,
    PreferenceDimension,
    RouteAttributes,
)
from src.recommendation import (
    PersonalizedRouteRanker,
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


class PersonalizedRouteRankerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ranker = PersonalizedRouteRanker()
        self.fast = RouteAttributes("fast", 30, 70, 300, 1)
        self.cheap = RouteAttributes("cheap", 90, 10, 300, 1)

    def test_time_and_cost_profiles_produce_different_recommendations(self) -> None:
        time_result = self.ranker.rank(
            (self.fast, self.cheap),
            _weights(time=0.8, cost=0.2),
        )
        cost_result = self.ranker.rank(
            (self.fast, self.cheap),
            _weights(time=0.2, cost=0.8),
        )

        self.assertEqual(time_result.recommended.route.route_id, "fast")
        self.assertEqual(cost_result.recommended.route.route_id, "cheap")

    def test_personalized_cost_is_sum_of_weighted_normalized_attributes(self) -> None:
        result = self.ranker.rank(
            (self.fast, self.cheap),
            _weights(time=2.0, cost=1.0),
        )

        fast = next(item for item in result.ranked_routes if item.route.route_id == "fast")
        expected = (2.0 / 3.0) * (30.0 / 180.0) + (1.0 / 3.0) * (70.0 / 100.0)
        self.assertAlmostEqual(fast.personalized_cost, expected)
        self.assertAlmostEqual(
            sum(fast.weighted_contributions.values()),
            fast.personalized_cost,
        )

    def test_hard_constraints_remove_unacceptable_routes_before_ranking(self) -> None:
        result = self.ranker.rank(
            (self.fast, self.cheap),
            _weights(time=0.9, cost=0.1),
            constraints=RouteConstraints(max_total_cost=50.0),
        )

        self.assertEqual(result.recommended.route.route_id, "cheap")
        self.assertEqual(len(result.rejected_routes), 1)
        self.assertEqual(result.rejected_routes[0].route.route_id, "fast")
        self.assertEqual(
            result.rejected_routes[0].violated_dimensions,
            (PreferenceDimension.COST,),
        )

    def test_no_feasible_route_is_a_valid_explainable_result(self) -> None:
        result = self.ranker.rank(
            (self.fast, self.cheap),
            _weights(time=0.5, cost=0.5),
            constraints=RouteConstraints(max_total_cost=1.0),
        )

        self.assertIsNone(result.recommended)
        self.assertEqual(result.feasible_count, 0)
        self.assertEqual(len(result.rejected_routes), 2)

    def test_top_k_limits_output_but_preserves_feasible_count(self) -> None:
        third = RouteAttributes("third", 120, 40, 800, 2)
        result = self.ranker.rank(
            (self.fast, self.cheap, third),
            _weights(time=0.5, cost=0.5),
            top_k=2,
        )

        self.assertEqual(len(result.ranked_routes), 2)
        self.assertEqual(result.feasible_count, 3)
        self.assertEqual([item.rank for item in result.ranked_routes], [1, 2])

    def test_top_one_advantages_are_compared_with_all_feasible_routes(self) -> None:
        third = RouteAttributes("third", 120, 40, 800, 2)
        result = self.ranker.rank(
            (self.fast, self.cheap, third),
            _weights(time=0.7, cost=0.2, walking=0.05, transfers=0.05),
            top_k=1,
        )

        self.assertTrue(result.recommended.advantage_dimensions)

    def test_favour_learning_result_can_be_used_directly(self) -> None:
        comparisons = tuple(
            PairwisePreference(
                chosen=RouteAttributes(f"chosen-{index}", 70, 8, 400, 1),
                rejected=RouteAttributes(f"rejected-{index}", 35, 55, 400, 1),
            )
            for index in range(6)
        )
        profile = PairwisePreferenceWeightLearner().fit(comparisons)

        result = self.ranker.rank((self.fast, self.cheap), profile)

        self.assertGreater(
            profile.weights[PreferenceDimension.COST],
            profile.weights[PreferenceDimension.TIME],
        )
        self.assertEqual(result.recommended.route.route_id, "cheap")

    def test_invalid_weights_top_k_and_duplicate_ids_are_rejected(self) -> None:
        incomplete = {dimension: 1.0 for dimension in PREFERENCE_DIMENSIONS[:-1]}
        with self.assertRaises(RecommendationValidationError):
            self.ranker.rank((self.fast,), incomplete)
        with self.assertRaises(RecommendationValidationError):
            self.ranker.rank((self.fast,), _weights(time=1.0, cost=1.0), top_k=0)
        duplicate = RouteAttributes("fast", 40, 30, 200, 0)
        with self.assertRaises(RecommendationValidationError):
            self.ranker.rank(
                (self.fast, duplicate),
                _weights(time=1.0, cost=1.0),
            )


if __name__ == "__main__":
    unittest.main()
