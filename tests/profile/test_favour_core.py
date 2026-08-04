"""FAVOUR 四维缩减实现的公式级和流程级验证。"""

from __future__ import annotations

from math import isclose
from math import pi, sqrt
import unittest

from src.profile import (
    BradleyTerryLogitLikelihood,
    FavourPosteriorObjective,
    FavourPosteriorPredictor,
    FixedGaussianPriorProvider,
    GaussianPreferencePrior,
    GroupPreferencePrior,
    InMemoryPosteriorRepository,
    MassPreferencePriorEstimator,
    NormalizedCostFeatureExtractor,
    PREFERENCE_DIMENSIONS,
    PairwisePreference,
    PairwisePreferenceWeightLearner,
    PreferenceDimension,
    PreferencePosterior,
    RouteAttributes,
)


def _cost_sensitive_comparison(suffix: str = "1") -> PairwisePreference:
    """选择更便宜但更慢的路线。"""

    return PairwisePreference(
        chosen=RouteAttributes(f"cheap-{suffix}", 60, 0, 300, 1),
        rejected=RouteAttributes(f"fast-{suffix}", 30, 100, 300, 1),
    )


def _posterior(
    coefficients: tuple[float, float, float, float],
    variance: float,
    evidence_count: int,
) -> PreferencePosterior:
    covariance = tuple(
        tuple(variance if row == column else 0.0 for column in range(4))
        for row in range(4)
    )
    values = {
        dimension: coefficients[index]
        for index, dimension in enumerate(PREFERENCE_DIMENSIONS)
    }
    return PreferencePosterior(
        coefficients=values,
        covariance=covariance,
        lower_bounds={dimension: 0.0 for dimension in PREFERENCE_DIMENSIONS},
        upper_bounds={dimension: 10.0 for dimension in PREFERENCE_DIMENSIONS},
        evidence_count=evidence_count,
        converged=True,
        iterations=2,
        negative_log_posterior=1.0,
        prior_name="test",
        prior_version="v1",
        feature_schema_version="four-cost-v1",
    )


class FavourFormulaTests(unittest.TestCase):
    """检查公式符号、导数和 Gaussian 后验计算。"""

    def test_logit_probability_is_symmetric(self) -> None:
        likelihood = BradleyTerryLogitLikelihood()
        self.assertAlmostEqual(
            likelihood.probability(1.7) + likelihood.probability(-1.7),
            1.0,
            places=12,
        )

    def test_analytic_gradient_matches_finite_difference(self) -> None:
        extractor = NormalizedCostFeatureExtractor()
        observation = extractor.extract_comparison(_cost_sensitive_comparison())
        prior = FixedGaussianPriorProvider.uniform(variance=2.0)
        objective = FavourPosteriorObjective((observation,), prior)
        point = (1.2, 0.9, 1.1, 0.8)
        analytic = objective.evaluate(point).gradient
        epsilon = 1e-6

        for index in range(4):
            left = list(point)
            right = list(point)
            left[index] -= epsilon
            right[index] += epsilon
            numeric = (
                objective.evaluate(tuple(right)).value
                - objective.evaluate(tuple(left)).value
            ) / (2.0 * epsilon)
            self.assertAlmostEqual(analytic[index], numeric, places=6)

    def test_observation_reduces_variance_along_cost_direction(self) -> None:
        prior = FixedGaussianPriorProvider.uniform(variance=5.0, upper_bound=50.0)
        result = PairwisePreferenceWeightLearner().fit(
            tuple(_cost_sensitive_comparison(str(index)) for index in range(12)),
            prior,
        )
        cost_index = PREFERENCE_DIMENSIONS.index(PreferenceDimension.COST)
        self.assertLess(result.posterior.covariance[cost_index][cost_index], 5.0)
        self.assertTrue(result.posterior.converged)

    def test_mass_preference_prior_matches_formula_seven(self) -> None:
        first = _posterior((1.0, 2.0, 3.0, 4.0), variance=1.0, evidence_count=2)
        second = _posterior((3.0, 4.0, 5.0, 6.0), variance=2.0, evidence_count=3)
        prior = MassPreferencePriorEstimator(
            covariance_regularization=1e-6
        ).estimate((first, second))

        self.assertEqual(prior.mean_vector(), (2.0, 3.0, 4.0, 5.0))
        self.assertAlmostEqual(prior.covariance[0][0], 2.500001, places=6)
        self.assertAlmostEqual(prior.covariance[0][1], 1.0, places=12)
        self.assertEqual(prior.evidence_count, 5)

    def test_posterior_prediction_matches_formula_nine(self) -> None:
        posterior = _posterior((2.0, 0.0, 0.0, 0.0), variance=1.0, evidence_count=2)
        comparison = PairwisePreference(
            chosen=RouteAttributes("chosen", 0, 10, 100, 0),
            rejected=RouteAttributes("rejected", 90, 10, 100, 0),
        )
        observation = NormalizedCostFeatureExtractor().extract_comparison(comparison)
        difference = observation.cost_difference()
        variance = difference[0] ** 2
        attenuation = 1.0 / sqrt(1.0 + pi * variance / 8.0)
        expected = BradleyTerryLogitLikelihood.probability(
            attenuation * 2.0 * difference[0]
        )

        actual = FavourPosteriorPredictor().probability_chosen(
            posterior,
            observation,
        )
        self.assertAlmostEqual(actual, expected, places=12)


class FavourWorkflowTests(unittest.TestCase):
    """验证四维画像、非单纯形系数、预测和增量状态。"""

    def test_no_evidence_returns_four_equal_profile_dimensions(self) -> None:
        result = PairwisePreferenceWeightLearner().fit((), GroupPreferencePrior.uniform())

        self.assertEqual(tuple(result.weights), PREFERENCE_DIMENSIONS)
        self.assertTrue(all(isclose(value, 0.25) for value in result.weights.values()))
        self.assertEqual(result.diagnostics.evidence_count, 0)
        self.assertEqual(sum(result.coefficients.values()), 4.0)

    def test_cost_evidence_changes_unconstrained_coefficients_and_display_weights(self) -> None:
        comparisons = tuple(
            _cost_sensitive_comparison(str(index)) for index in range(20)
        )
        result = PairwisePreferenceWeightLearner().fit(
            comparisons,
            GroupPreferencePrior.uniform(),
        )

        self.assertEqual(set(result.weights), set(PREFERENCE_DIMENSIONS))
        self.assertAlmostEqual(sum(result.weights.values()), 1.0, places=12)
        self.assertFalse(isclose(sum(result.coefficients.values()), 1.0))
        self.assertEqual(
            max(PREFERENCE_DIMENSIONS, key=result.weights.__getitem__),
            PreferenceDimension.COST,
        )
        self.assertEqual(result.diagnostics.evidence_count, len(comparisons))

    def test_each_of_the_four_profile_dimensions_can_be_recovered(self) -> None:
        route_values = {
            PreferenceDimension.TIME: (20, 20, 200, 0),
            PreferenceDimension.COST: (60, 0, 200, 0),
            PreferenceDimension.WALKING_DISTANCE: (60, 20, 0, 0),
            PreferenceDimension.TRANSFERS: (60, 20, 200, 0),
        }
        rejected_values = {
            PreferenceDimension.TIME: (180, 20, 200, 0),
            PreferenceDimension.COST: (60, 100, 200, 0),
            PreferenceDimension.WALKING_DISTANCE: (60, 20, 3000, 0),
            PreferenceDimension.TRANSFERS: (60, 20, 200, 4),
        }

        for dimension in PREFERENCE_DIMENSIONS:
            chosen = RouteAttributes(
                f"chosen-{dimension.value}",
                *route_values[dimension],
            )
            rejected = RouteAttributes(
                f"rejected-{dimension.value}",
                *rejected_values[dimension],
            )
            evidence = tuple(
                PairwisePreference(chosen, rejected) for _ in range(12)
            )
            result = PairwisePreferenceWeightLearner().fit(
                evidence,
                GroupPreferencePrior.uniform(),
            )

            with self.subTest(dimension=dimension):
                self.assertEqual(
                    max(PREFERENCE_DIMENSIONS, key=result.weights.__getitem__),
                    dimension,
                )

    def test_strong_evidence_is_not_capped_by_simplex_probability(self) -> None:
        comparisons = tuple(
            _cost_sensitive_comparison(str(index)) for index in range(80)
        )
        prior = FixedGaussianPriorProvider.uniform(variance=25.0, upper_bound=50.0)
        result = PairwisePreferenceWeightLearner().fit(comparisons, prior)
        extractor = NormalizedCostFeatureExtractor()
        observation = extractor.extract_comparison(comparisons[0])
        likelihood_margin = sum(
            coefficient * delta
            for coefficient, delta in zip(
                result.posterior.coefficient_vector(),
                observation.cost_difference(),
                strict=True,
            )
        )

        self.assertGreater(BradleyTerryLogitLikelihood.probability(likelihood_margin), 0.90)

    def test_incremental_service_persists_and_predicts(self) -> None:
        repository = InMemoryPosteriorRepository()
        service = PairwisePreferenceWeightLearner().create_service(
            GroupPreferencePrior.uniform(),
            repository,
        )
        comparison = _cost_sensitive_comparison()

        first = service.update_user("user-1", comparison)
        second = service.update_user("user-1", comparison)
        probability = service.predict_user_choice("user-1", comparison)

        self.assertEqual(first.diagnostics.evidence_count, 1)
        self.assertEqual(second.diagnostics.evidence_count, 2)
        self.assertGreater(probability, 0.5)
        self.assertLessEqual(probability, 1.0)


if __name__ == "__main__":
    unittest.main()
