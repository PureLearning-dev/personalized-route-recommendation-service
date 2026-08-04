"""FAVOUR Gaussian prior、Logit 似然和 Laplace 后验推断。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import pi, sqrt

from .exceptions import ProfileValidationError
from .models import (
    FeatureComparison,
    GaussianPreferencePrior,
    Matrix,
    PREFERENCE_DIMENSIONS,
    PreferencePosterior,
)
from .optimization import (
    BoxConstrainedNewtonOptimizer,
    ObjectiveEvaluation,
    Vector,
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
    """一次成对选择对目标、梯度和 Hessian 的贡献。"""

    negative_log_likelihood: float
    gradient_factor: float
    curvature: float
    probability: float


class BradleyTerryLogitLikelihood:
    """实现 FAVOUR 公式（4）（5）的稳定 Logit 数值。"""

    @staticmethod
    def probability(margin: float) -> float:
        return sigmoid(margin)

    def evaluate(self, margin: float, evidence_weight: float) -> LogitLikelihoodTerms:
        probability = self.probability(margin)
        return LogitLikelihoodTerms(
            negative_log_likelihood=evidence_weight * softplus(-margin),
            gradient_factor=evidence_weight * (probability - 1.0),
            curvature=evidence_weight * probability * (1.0 - probability),
            probability=probability,
        )


class FavourPosteriorObjective:
    """负对数后验及其解析梯度、Hessian。"""

    def __init__(
        self,
        observations: Sequence[FeatureComparison],
        prior: GaussianPreferencePrior,
        likelihood: BradleyTerryLogitLikelihood | None = None,
    ) -> None:
        if any(
            observation.schema_version != prior.feature_schema_version
            for observation in observations
        ):
            raise ProfileValidationError("路线特征版本与先验版本不一致")
        self._observations = tuple(observations)
        self._prior = prior
        self._prior_mean = prior.mean_vector()
        self._precision = inverse_matrix(prior.covariance)
        self._likelihood = likelihood or BradleyTerryLogitLikelihood()

    def evaluate(self, coefficients: Vector) -> ObjectiveEvaluation:
        size = len(PREFERENCE_DIMENSIONS)
        centered = tuple(
            current - prior
            for current, prior in zip(coefficients, self._prior_mean, strict=True)
        )
        precision_centered = matrix_vector_product(self._precision, centered)
        value = 0.5 * quadratic_form(centered, self._precision)
        gradient = list(precision_centered)
        hessian = [list(row) for row in self._precision]

        for observation in self._observations:
            difference = observation.cost_difference()
            margin = dot(coefficients, difference)
            terms = self._likelihood.evaluate(margin, observation.evidence_weight)
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


class FavourLaplacePosteriorEstimator:
    """求后验众数并以众数处 Hessian 的逆作为 Gaussian 协方差。"""

    def __init__(
        self,
        optimizer: BoxConstrainedNewtonOptimizer | None = None,
        likelihood: BradleyTerryLogitLikelihood | None = None,
    ) -> None:
        self._optimizer = optimizer or BoxConstrainedNewtonOptimizer()
        self._likelihood = likelihood or BradleyTerryLogitLikelihood()

    def fit(
        self,
        observations: Sequence[FeatureComparison],
        prior: GaussianPreferencePrior,
    ) -> PreferencePosterior:
        observations = tuple(observations)
        if not observations:
            return PreferencePosterior(
                coefficients=dict(prior.mean),
                covariance=prior.covariance,
                lower_bounds=prior.lower_bounds,
                upper_bounds=prior.upper_bounds,
                evidence_count=prior.evidence_count,
                converged=True,
                iterations=0,
                negative_log_posterior=0.0,
                prior_name=prior.name,
                prior_version=prior.version,
                feature_schema_version=prior.feature_schema_version,
            )

        objective = FavourPosteriorObjective(observations, prior, self._likelihood)
        optimized = self._optimizer.optimize(
            objective.evaluate,
            prior.mean_vector(),
            prior.lower_bound_vector(),
            prior.upper_bound_vector(),
        )
        covariance: Matrix = inverse_matrix(optimized.evaluation.hessian, initial_jitter=1e-12)
        coefficients = {
            dimension: optimized.point[index]
            for index, dimension in enumerate(PREFERENCE_DIMENSIONS)
        }
        return PreferencePosterior(
            coefficients=coefficients,
            covariance=covariance,
            lower_bounds=prior.lower_bounds,
            upper_bounds=prior.upper_bounds,
            evidence_count=prior.evidence_count + len(observations),
            converged=optimized.converged,
            iterations=optimized.iterations,
            negative_log_posterior=optimized.evaluation.value,
            prior_name=prior.name,
            prior_version=prior.version,
            feature_schema_version=prior.feature_schema_version,
        )

    def update(
        self,
        posterior: PreferencePosterior,
        observation: FeatureComparison,
    ) -> PreferencePosterior:
        """使用上一次 Gaussian 后验作为新选择的先验。"""

        return self.fit((observation,), posterior.as_incremental_prior())


class FavourPosteriorPredictor:
    """使用 FAVOUR 公式（9）进行考虑后验不确定性的选择预测。"""

    def probability_chosen(
        self,
        posterior: PreferencePosterior,
        observation: FeatureComparison,
    ) -> float:
        if posterior.feature_schema_version != observation.schema_version:
            raise ProfileValidationError("后验与待预测路线的特征版本不一致")
        difference = observation.cost_difference()
        variance = max(0.0, quadratic_form(difference, posterior.covariance))
        attenuation = 1.0 / sqrt(1.0 + pi * variance / 8.0)
        margin = attenuation * dot(posterior.coefficient_vector(), difference)
        return sigmoid(margin)
