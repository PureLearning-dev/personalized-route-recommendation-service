"""严格FAVOUR四维实现的公式、优化器和流程验证。"""

from __future__ import annotations

from math import isclose, log, pi, sqrt
import unittest
from unittest.mock import patch

from src.profile import (
    BradleyTerryLogitLikelihood,
    FavourPosteriorObjective,
    FavourPosteriorPredictor,
    GaussianPreferenceModel,
    MassPreferencePriorEstimator,
    NormalizedCostFeatureExtractor,
    PREFERENCE_DIMENSIONS,
    PairwisePreference,
    PairwisePreferenceWeightLearner,
    PreferenceDimension,
    PreferencePreset,
    ProfileNumericalError,
    ProfileValidationError,
    RouteAttributes,
    preference_prior_from_weights,
    preset_preference_weights,
    standard_mass_preference_prior,
)
from src.profile.optimization import (
    BoxBoundedTrustRegionOptimizer,
    ObjectiveEvaluation,
)
from src.profile.inference import FavourLaplaceInference


def _cost_sensitive_comparison(suffix: str = "1") -> PairwisePreference:
    return PairwisePreference(
        chosen=RouteAttributes(f"cheap-{suffix}", 60, 0, 300, 1),
        rejected=RouteAttributes(f"fast-{suffix}", 30, 100, 300, 1),
    )


def _model(
    coefficients: tuple[float, float, float, float],
    variance: float,
) -> GaussianPreferenceModel:
    covariance = tuple(
        tuple(variance if row == column else 0.0 for column in range(4))
        for row in range(4)
    )
    return GaussianPreferenceModel(
        mean={
            dimension: coefficients[index]
            for index, dimension in enumerate(PREFERENCE_DIMENSIONS)
        },
        covariance=covariance,
        lower_bounds={dimension: -20.0 for dimension in PREFERENCE_DIMENSIONS},
        upper_bounds={dimension: 0.0 for dimension in PREFERENCE_DIMENSIONS},
    )


class FavourFormulaTests(unittest.TestCase):
    def test_formula_four_logit_probability_is_symmetric(self) -> None:
        likelihood = BradleyTerryLogitLikelihood()
        self.assertAlmostEqual(
            likelihood.probability(1.7) + likelihood.probability(-1.7),
            1.0,
            places=12,
        )

    def test_formula_five_joint_likelihood_matches_probability_product(self) -> None:
        extractor = NormalizedCostFeatureExtractor()
        observations = (
            extractor.extract_comparison(_cost_sensitive_comparison("first")),
            extractor.extract_comparison(_cost_sensitive_comparison("second")),
        )
        point = (-1.0, -2.0, -1.0, -1.0)
        objective = FavourPosteriorObjective(
            observations,
            standard_mass_preference_prior(),
        )
        probability_product = 1.0
        for observation in observations:
            margin = sum(
                coefficient * difference
                for coefficient, difference in zip(
                    point,
                    observation.utility_difference(),
                    strict=True,
                )
            )
            probability_product *= BradleyTerryLogitLikelihood.probability(margin)

        gaussian_penalty = 0.5 * sum(value * value for value in point)
        self.assertAlmostEqual(
            objective.evaluate(point).value,
            gaussian_penalty - log(probability_product),
            places=12,
        )

    def test_analytic_gradient_matches_finite_difference(self) -> None:
        observation = NormalizedCostFeatureExtractor().extract_comparison(
            _cost_sensitive_comparison()
        )
        objective = FavourPosteriorObjective(
            (observation,),
            standard_mass_preference_prior(),
        )
        point = (-1.2, -0.9, -1.1, -0.8)
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

    def test_analytic_hessian_matches_finite_difference(self) -> None:
        observation = NormalizedCostFeatureExtractor().extract_comparison(
            _cost_sensitive_comparison()
        )
        objective = FavourPosteriorObjective(
            (observation,),
            standard_mass_preference_prior(),
        )
        point = (-1.2, -0.9, -1.1, -0.8)
        analytic = objective.evaluate(point).hessian
        epsilon = 1e-6

        for column in range(4):
            left = list(point)
            right = list(point)
            left[column] -= epsilon
            right[column] += epsilon
            left_gradient = objective.evaluate(tuple(left)).gradient
            right_gradient = objective.evaluate(tuple(right)).gradient
            for row in range(4):
                numeric = (right_gradient[row] - left_gradient[row]) / (2.0 * epsilon)
                self.assertAlmostEqual(analytic[row][column], numeric, places=6)

    def test_formula_seven_mass_preference_aggregation(self) -> None:
        first = _model((-1.0, -2.0, -3.0, -4.0), 1.0)
        second = _model((-3.0, -4.0, -5.0, -6.0), 2.0)
        aggregated = MassPreferencePriorEstimator.aggregate((first, second))

        self.assertEqual(aggregated.mean_vector(), (-2.0, -3.0, -4.0, -5.0))
        self.assertAlmostEqual(aggregated.covariance[0][0], 2.5)
        self.assertAlmostEqual(aggregated.covariance[0][1], 1.0)

    def test_formula_nine_prediction_matches_manual_value(self) -> None:
        posterior = _model((-2.0, 0.0, 0.0, 0.0), 1.0)
        comparison = PairwisePreference(
            chosen=RouteAttributes("chosen", 0, 10, 100, 0),
            rejected=RouteAttributes("rejected", 90, 10, 100, 0),
        )
        observation = NormalizedCostFeatureExtractor().extract_comparison(comparison)
        difference = observation.utility_difference()
        variance = difference[0] ** 2
        attenuation = 1.0 / sqrt(1.0 + pi * variance / 8.0)
        expected = BradleyTerryLogitLikelihood.probability(
            attenuation * -2.0 * difference[0]
        )

        self.assertAlmostEqual(
            FavourPosteriorPredictor.probability(posterior, observation),
            expected,
            places=12,
        )


class TrustRegionTests(unittest.TestCase):
    def test_box_bounded_trust_region_finds_quadratic_minimum(self) -> None:
        target = (-2.0, -3.0, -4.0, -5.0)
        identity = tuple(
            tuple(1.0 if row == column else 0.0 for column in range(4))
            for row in range(4)
        )

        def objective(point: tuple[float, ...]) -> ObjectiveEvaluation:
            difference = tuple(
                value - expected
                for value, expected in zip(point, target, strict=True)
            )
            return ObjectiveEvaluation(
                value=0.5 * sum(value * value for value in difference),
                gradient=difference,
                hessian=identity,
            )

        result = BoxBoundedTrustRegionOptimizer().optimize(
            objective,
            (-20.0,) * 4,
            (0.0,) * 4,
        )

        self.assertTrue(result.converged)
        for actual, expected in zip(result.point, target, strict=True):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_box_bounded_trust_region_handles_coupled_active_bound(self) -> None:
        """边界变量不能通过 Hessian 耦合项干扰其余自由变量。"""

        hessian = ((2.0, 1.0), (1.0, 2.0))
        unconstrained_target = (1.0, -2.0)

        def objective(point: tuple[float, ...]) -> ObjectiveEvaluation:
            difference = tuple(
                value - target
                for value, target in zip(point, unconstrained_target, strict=True)
            )
            gradient = tuple(
                sum(row[column] * difference[column] for column in range(2))
                for row in hessian
            )
            return ObjectiveEvaluation(
                value=0.5
                * sum(
                    difference[row] * gradient[row]
                    for row in range(2)
                ),
                gradient=gradient,
                hessian=hessian,
            )

        result = BoxBoundedTrustRegionOptimizer().optimize(
            objective,
            (-5.0, -5.0),
            (0.0, 0.0),
        )

        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.point[0], 0.0, places=7)
        self.assertAlmostEqual(result.point[1], -1.5, places=7)

    def test_optimizer_rejects_result_when_no_start_converges(self) -> None:
        """达到迭代上限的随机起点不能被当作可用最优解返回。"""

        identity = tuple(
            tuple(1.0 if row == column else 0.0 for column in range(4))
            for row in range(4)
        )

        def objective(point: tuple[float, ...]) -> ObjectiveEvaluation:
            return ObjectiveEvaluation(
                value=0.5 * sum(value * value for value in point),
                gradient=point,
                hessian=identity,
            )

        with patch.object(BoxBoundedTrustRegionOptimizer, "_MAX_ITERATIONS", 1):
            with self.assertRaises(ProfileNumericalError):
                BoxBoundedTrustRegionOptimizer().optimize(
                    objective,
                    (-20.0,) * 4,
                    (0.0,) * 4,
                )

    def test_optimizer_rejects_invalid_bounds(self) -> None:
        """无效边界应在进入随机起点计算前给出领域错误。"""

        with self.assertRaises(ProfileValidationError):
            BoxBoundedTrustRegionOptimizer().optimize(
                lambda point: ObjectiveEvaluation(0.0, point, ((1.0,),)),
                (1.0,),
                (0.0,),
            )


class EngineeringGuardTests(unittest.TestCase):
    def test_gaussian_covariance_must_be_positive_definite(self) -> None:
        """正对角线和对称性不足以证明矩阵是合法协方差。"""

        indefinite = (
            (1.0, 2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        with self.assertRaises(ProfileValidationError):
            GaussianPreferenceModel(
                mean={dimension: 0.0 for dimension in PREFERENCE_DIMENSIONS},
                covariance=indefinite,
                lower_bounds={
                    dimension: -20.0 for dimension in PREFERENCE_DIMENSIONS
                },
                upper_bounds={dimension: 0.0 for dimension in PREFERENCE_DIMENSIONS},
            )

    def test_normalization_preserves_values_above_reference_scale(self) -> None:
        """超出参考尺度的路线仍应保留差异，避免真实路线排序失真。"""

        comparison = PairwisePreference(
            chosen=RouteAttributes("long", 360, 10, 100, 0),
            rejected=RouteAttributes("very-long", 720, 10, 100, 0),
        )
        observation = NormalizedCostFeatureExtractor().extract_comparison(comparison)

        self.assertEqual(observation.chosen[0], 2.0)
        self.assertEqual(observation.rejected[0], 4.0)
        self.assertNotEqual(observation.chosen[0], observation.rejected[0])

    def test_mpp_rejects_empty_history_for_any_user(self) -> None:
        """空历史不能作为一个有效用户参与群体先验估计。"""

        estimator = MassPreferencePriorEstimator(FavourLaplaceInference())
        with self.assertRaises(ProfileValidationError):
            estimator.refine(((),), standard_mass_preference_prior())

    def test_mpp_refinement_has_a_hard_iteration_limit(self) -> None:
        """持续漂移的MPP必须终止并报告错误，不能永久占用服务线程。"""

        class DriftingInference:
            def infer(
                self,
                observations: tuple[object, ...],
                prior: GaussianPreferenceModel,
            ) -> tuple[GaussianPreferenceModel, bool]:
                del observations
                shifted = {
                    dimension: prior.mean[dimension] - 0.25
                    for dimension in PREFERENCE_DIMENSIONS
                }
                return (
                    GaussianPreferenceModel(
                        mean=shifted,
                        covariance=prior.covariance,
                        lower_bounds=prior.lower_bounds,
                        upper_bounds=prior.upper_bounds,
                    ),
                    True,
                )

        estimator = MassPreferencePriorEstimator(DriftingInference())  # type: ignore[arg-type]
        with patch.object(MassPreferencePriorEstimator, "_MAX_ITERATIONS", 2):
            with self.assertRaises(ProfileNumericalError):
                estimator.refine(
                    ((object(),),),
                    standard_mass_preference_prior(),
                )


class FavourWorkflowTests(unittest.TestCase):
    def test_no_evidence_returns_paper_mpp_initialization(self) -> None:
        result = PairwisePreferenceWeightLearner().fit(())

        self.assertEqual(result.posterior.mean_vector(), (0.0, 0.0, 0.0, 0.0))
        self.assertTrue(all(isclose(value, 0.25) for value in result.weights.values()))
        self.assertEqual(result.evidence_count, 0)

    def test_formula_six_incremental_learning_raises_cost_sensitivity(self) -> None:
        comparisons = tuple(
            _cost_sensitive_comparison(str(index)) for index in range(12)
        )
        result = PairwisePreferenceWeightLearner().fit(comparisons)

        self.assertTrue(result.converged)
        self.assertEqual(result.evidence_count, len(comparisons))
        self.assertTrue(all(value <= 0.0 for value in result.utility_coefficients.values()))
        self.assertEqual(
            max(PREFERENCE_DIMENSIONS, key=result.weights.__getitem__),
            PreferenceDimension.COST,
        )
        self.assertGreater(sum(result.choice_probabilities) / len(comparisons), 0.5)

    def test_named_presets_replace_the_equal_weight_fallback(self) -> None:
        expected_dominants = {
            PreferencePreset.TIME_PRIORITY: PreferenceDimension.TIME,
            PreferencePreset.COST_PRIORITY: PreferenceDimension.COST,
            PreferencePreset.LOW_WALKING: PreferenceDimension.WALKING_DISTANCE,
            PreferencePreset.LOW_TRANSFERS: PreferenceDimension.TRANSFERS,
        }

        for preset, dominant in expected_dominants.items():
            result = PairwisePreferenceWeightLearner().fit(
                (),
                preference_preset=preset,
            )

            with self.subTest(preset=preset):
                self.assertAlmostEqual(result.weights[dominant], 0.70)
                self.assertEqual(
                    max(PREFERENCE_DIMENSIONS, key=result.weights.__getitem__),
                    dominant,
                )

    def test_custom_percentage_weights_are_normalized_into_a_gaussian_prior(self) -> None:
        weights = {
            PreferenceDimension.TIME: 55.0,
            PreferenceDimension.COST: 25.0,
            PreferenceDimension.WALKING_DISTANCE: 15.0,
            PreferenceDimension.TRANSFERS: 5.0,
        }
        prior = preference_prior_from_weights(weights)
        sensitivities = {
            dimension: -prior.mean[dimension]
            for dimension in PREFERENCE_DIMENSIONS
        }
        total = sum(sensitivities.values())

        for dimension in PREFERENCE_DIMENSIONS:
            self.assertAlmostEqual(
                sensitivities[dimension] / total,
                weights[dimension] / 100,
            )

    def test_preset_profile_can_be_updated_by_route_choices(self) -> None:
        initial = PairwisePreferenceWeightLearner().fit(
            (),
            preference_preset=PreferencePreset.TIME_PRIORITY,
        )
        comparisons = tuple(
            _cost_sensitive_comparison(f"preset-update-{index}")
            for index in range(24)
        )
        updated = PairwisePreferenceWeightLearner().fit(
            comparisons,
            preference_preset=PreferencePreset.TIME_PRIORITY,
        )

        self.assertEqual(updated.evidence_count, len(comparisons))
        self.assertGreater(
            updated.weights[PreferenceDimension.COST],
            initial.weights[PreferenceDimension.COST],
        )
        self.assertEqual(
            max(PREFERENCE_DIMENSIONS, key=updated.weights.__getitem__),
            PreferenceDimension.COST,
        )

    def test_preset_weight_copy_does_not_mutate_global_configuration(self) -> None:
        weights = preset_preference_weights(PreferencePreset.COST_PRIORITY)
        weights[PreferenceDimension.COST] = 0.0

        reloaded = preset_preference_weights(PreferencePreset.COST_PRIORITY)
        self.assertAlmostEqual(reloaded[PreferenceDimension.COST], 0.70)

    def test_each_of_the_four_dimensions_can_be_recovered(self) -> None:
        chosen_values = {
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
                *chosen_values[dimension],
            )
            rejected = RouteAttributes(
                f"rejected-{dimension.value}",
                *rejected_values[dimension],
            )
            comparisons = tuple(
                PairwisePreference(chosen, rejected) for _ in range(10)
            )
            result = PairwisePreferenceWeightLearner().fit(comparisons)

            with self.subTest(dimension=dimension):
                self.assertEqual(
                    max(PREFERENCE_DIMENSIONS, key=result.weights.__getitem__),
                    dimension,
                )

    def test_group_histories_refine_the_mpp_used_for_a_new_user(self) -> None:
        history = tuple(
            _cost_sensitive_comparison(f"history-{index}") for index in range(4)
        )
        result = PairwisePreferenceWeightLearner().fit(
            (),
            group_histories=(history, history),
        )

        self.assertEqual(
            max(PREFERENCE_DIMENSIONS, key=result.weights.__getitem__),
            PreferenceDimension.COST,
        )


if __name__ == "__main__":
    unittest.main()
