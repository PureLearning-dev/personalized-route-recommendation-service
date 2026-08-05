"""FAVOUR公式（4）至（9）的最小推断实现。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import log, pi, sqrt

from .exceptions import ProfileValidationError
from .models import (
    FeatureComparison,
    GaussianPreferenceModel,
    Matrix,
    PREFERENCE_DIMENSIONS,
)
from .optimization import (
    BoxBoundedTrustRegionOptimizer,
    ObjectiveEvaluation,
    Vector,
    determinant,
    dot,
    inverse_matrix,
    matrix_vector_product,
    quadratic_form,
    sigmoid,
    softplus,
    symmetrize,
)


@dataclass(frozen=True, slots=True)
class LogitLikelihoodTerms:
    negative_log_likelihood: float
    gradient_factor: float
    curvature: float


class BradleyTerryLogitLikelihood:
    """论文公式（4）（5）的二元Logit似然。"""

    @staticmethod
    def probability(margin: float) -> float:
        return sigmoid(margin)

    def evaluate(self, margin: float) -> LogitLikelihoodTerms:
        probability = self.probability(margin)
        return LogitLikelihoodTerms(
            negative_log_likelihood=softplus(-margin),
            gradient_factor=probability - 1.0,
            curvature=probability * (1.0 - probability),
        )


class FavourPosteriorObjective:
    """Gaussian先验与公式（5）联合似然形成的负对数后验。"""

    def __init__(
        self,
        observations: Sequence[FeatureComparison],
        prior: GaussianPreferenceModel,
    ) -> None:
        self._observations = tuple(observations)
        self._prior_mean = prior.mean_vector()
        self._precision = inverse_matrix(prior.covariance)
        self._likelihood = BradleyTerryLogitLikelihood()

    def evaluate(self, coefficients: Vector) -> ObjectiveEvaluation:
        size = len(PREFERENCE_DIMENSIONS)
        centered = tuple(
            current - prior
            for current, prior in zip(coefficients, self._prior_mean, strict=True)
        )
        gradient = list(matrix_vector_product(self._precision, centered))
        hessian = [list(row) for row in self._precision]
        value = 0.5 * quadratic_form(centered, self._precision)

        for observation in self._observations:
            difference = observation.utility_difference()
            terms = self._likelihood.evaluate(dot(coefficients, difference))
            value += terms.negative_log_likelihood
            for row in range(size):
                gradient[row] += terms.gradient_factor * difference[row]
                for column in range(size):
                    hessian[row][column] += (
                        terms.curvature * difference[row] * difference[column]
                    )

        return ObjectiveEvaluation(
            value=value,
            gradient=tuple(gradient),
            hessian=symmetrize(tuple(tuple(row) for row in hessian)),
        )


class FavourLaplaceInference:
    """论文的box-bounded后验众数与Laplace Gaussian近似。"""

    def __init__(self) -> None:
        self._optimizer = BoxBoundedTrustRegionOptimizer()

    def infer(
        self,
        observations: Sequence[FeatureComparison],
        prior: GaussianPreferenceModel,
    ) -> tuple[GaussianPreferenceModel, bool]:
        observations = tuple(observations)
        if not observations:
            return prior, True

        objective = FavourPosteriorObjective(observations, prior)
        optimized = self._optimizer.optimize(
            objective.evaluate,
            prior.lower_bound_vector(),
            prior.upper_bound_vector(),
        )
        covariance = inverse_matrix(optimized.evaluation.hessian)
        posterior = GaussianPreferenceModel(
            mean={
                dimension: optimized.point[index]
                for index, dimension in enumerate(PREFERENCE_DIMENSIONS)
            },
            covariance=covariance,
            lower_bounds=prior.lower_bounds,
            upper_bounds=prior.upper_bounds,
        )
        return posterior, optimized.converged

    def update_incrementally(
        self,
        observations: Sequence[FeatureComparison],
        prior: GaussianPreferenceModel,
    ) -> tuple[GaussianPreferenceModel, bool]:
        """按论文公式（6）把每次后验作为下一次先验。"""

        posterior = prior
        converged = True
        for observation in observations:
            posterior, step_converged = self.infer((observation,), posterior)
            converged = converged and step_converged
        return posterior, converged


class MassPreferencePriorEstimator:
    """按论文图2迭代精炼MPP，并用公式（7）聚合个人后验。"""

    _KL_TOLERANCE = 1e-3

    def __init__(self, inference: FavourLaplaceInference) -> None:
        self._inference = inference

    @staticmethod
    def aggregate(
        posteriors: Sequence[GaussianPreferenceModel],
    ) -> GaussianPreferenceModel:
        posteriors = tuple(posteriors)
        if not posteriors:
            raise ProfileValidationError("MPP聚合至少需要一个个人后验")

        vectors = [posterior.mean_vector() for posterior in posteriors]
        count = len(vectors)
        size = len(PREFERENCE_DIMENSIONS)
        mean = tuple(
            sum(vector[index] for vector in vectors) / count
            for index in range(size)
        )
        covariance: Matrix = tuple(
            tuple(
                sum(
                    posterior.covariance[row][column]
                    + (vector[row] - mean[row]) * (vector[column] - mean[column])
                    for posterior, vector in zip(posteriors, vectors, strict=True)
                )
                / count
                for column in range(size)
            )
            for row in range(size)
        )
        first = posteriors[0]
        return GaussianPreferenceModel(
            mean={
                dimension: mean[index]
                for index, dimension in enumerate(PREFERENCE_DIMENSIONS)
            },
            covariance=covariance,
            lower_bounds=first.lower_bounds,
            upper_bounds=first.upper_bounds,
        )

    @staticmethod
    def _kl_divergence(
        previous: GaussianPreferenceModel,
        current: GaussianPreferenceModel,
    ) -> float:
        inverse_current = inverse_matrix(current.covariance)
        mean_difference = tuple(
            current_value - previous_value
            for previous_value, current_value in zip(
                previous.mean_vector(),
                current.mean_vector(),
                strict=True,
            )
        )
        trace_term = sum(
            inverse_current[row][column] * previous.covariance[column][row]
            for row in range(len(PREFERENCE_DIMENSIONS))
            for column in range(len(PREFERENCE_DIMENSIONS))
        )
        determinant_ratio = determinant(current.covariance) / determinant(
            previous.covariance
        )
        return 0.5 * (
            trace_term
            + quadratic_form(mean_difference, inverse_current)
            - len(PREFERENCE_DIMENSIONS)
            + log(determinant_ratio)
        )

    def refine(
        self,
        user_training_sets: Sequence[Sequence[FeatureComparison]],
        initial_mpp: GaussianPreferenceModel,
    ) -> GaussianPreferenceModel:
        training_sets = tuple(tuple(items) for items in user_training_sets)
        if not training_sets:
            raise ProfileValidationError("MPP精炼至少需要一个历史用户")

        current = initial_mpp
        while True:
            posteriors = tuple(
                self._inference.infer(training_set, current)[0]
                for training_set in training_sets
            )
            refined = self.aggregate(posteriors)
            if self._kl_divergence(current, refined) < self._KL_TOLERANCE:
                return refined
            current = refined


class FavourPosteriorPredictor:
    """论文公式（9）的后验选择概率。"""

    @staticmethod
    def probability(
        posterior: GaussianPreferenceModel,
        observation: FeatureComparison,
    ) -> float:
        difference = observation.utility_difference()
        variance = max(0.0, quadratic_form(difference, posterior.covariance))
        attenuation = 1.0 / sqrt(1.0 + pi * variance / 8.0)
        return sigmoid(
            attenuation * dot(posterior.mean_vector(), difference)
        )
