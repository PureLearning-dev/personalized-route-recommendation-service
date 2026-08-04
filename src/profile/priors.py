"""固定 Gaussian 先验与 FAVOUR 公式（7）的 MPP 估计。"""

from __future__ import annotations

from collections.abc import Sequence

from .exceptions import ProfileValidationError
from .models import (
    GaussianPreferencePrior,
    GroupPreferencePrior,
    Matrix,
    PREFERENCE_DIMENSIONS,
    PreferenceDimension,
    PreferencePosterior,
)


def _diagonal_matrix(diagonal: Sequence[float]) -> Matrix:
    size = len(diagonal)
    return tuple(
        tuple(float(diagonal[row]) if row == column else 0.0 for column in range(size))
        for row in range(size)
    )


class FixedGaussianPriorProvider:
    """返回一个经过版本化的固定 Gaussian 冷启动先验。"""

    def __init__(self, prior: GaussianPreferencePrior | None = None) -> None:
        self._prior = prior or self.uniform()

    @staticmethod
    def uniform(
        coefficient_mean: float = 1.0,
        variance: float = 1.0,
        upper_bound: float = 20.0,
        feature_schema_version: str = "four-cost-v1",
    ) -> GaussianPreferencePrior:
        mean = {dimension: coefficient_mean for dimension in PREFERENCE_DIMENSIONS}
        bounds_lower = {dimension: 0.0 for dimension in PREFERENCE_DIMENSIONS}
        bounds_upper = {dimension: upper_bound for dimension in PREFERENCE_DIMENSIONS}
        return GaussianPreferencePrior(
            mean=mean,
            covariance=_diagonal_matrix([variance] * len(PREFERENCE_DIMENSIONS)),
            lower_bounds=bounds_lower,
            upper_bounds=bounds_upper,
            name="fixed-uniform-gaussian",
            version="v1",
            feature_schema_version=feature_schema_version,
        )

    def get_prior(self) -> GaussianPreferencePrior:
        return self._prior


class LegacyGroupPriorAdapter:
    """把旧版相对权重转换为可供 FAVOUR 使用的 Gaussian 先验。"""

    def __init__(
        self,
        coefficient_scale: float = 4.0,
        base_variance: float = 4.0,
        upper_bound: float = 20.0,
        feature_schema_version: str = "four-cost-v1",
    ) -> None:
        if coefficient_scale <= 0 or base_variance <= 0 or upper_bound <= 0:
            raise ProfileValidationError("旧先验适配参数必须为正数")
        self._coefficient_scale = float(coefficient_scale)
        self._base_variance = float(base_variance)
        self._upper_bound = float(upper_bound)
        self._feature_schema_version = feature_schema_version

    def convert(self, prior: GroupPreferencePrior) -> GaussianPreferencePrior:
        mean = {
            dimension: prior.weights[dimension] * self._coefficient_scale
            for dimension in PREFERENCE_DIMENSIONS
        }
        effective_strength = max(prior.equivalent_sample_size, 1.0)
        variance = self._base_variance / effective_strength
        return GaussianPreferencePrior(
            mean=mean,
            covariance=_diagonal_matrix([variance] * len(PREFERENCE_DIMENSIONS)),
            lower_bounds={dimension: 0.0 for dimension in PREFERENCE_DIMENSIONS},
            upper_bounds={
                dimension: self._upper_bound for dimension in PREFERENCE_DIMENSIONS
            },
            name=f"legacy:{prior.name}",
            version="legacy-adapter-v1",
            feature_schema_version=self._feature_schema_version,
        )


class MassPreferencePriorEstimator:
    """按 FAVOUR 公式（7）聚合多个用户的 Gaussian 后验。"""

    def __init__(self, covariance_regularization: float = 1e-6) -> None:
        if covariance_regularization <= 0:
            raise ProfileValidationError("MPP 协方差正则必须为正数")
        self._regularization = float(covariance_regularization)

    def estimate(
        self,
        posteriors: Sequence[PreferencePosterior],
        *,
        name: str = "mass-preference-prior",
        version: str = "mpp-v1",
    ) -> GaussianPreferencePrior:
        if not posteriors:
            raise ProfileValidationError("估计 MPP 至少需要一个用户后验")

        schema_version = posteriors[0].feature_schema_version
        if any(posterior.feature_schema_version != schema_version for posterior in posteriors):
            raise ProfileValidationError("参与 MPP 的后验必须使用同一特征版本")

        coefficient_vectors = [posterior.coefficient_vector() for posterior in posteriors]
        count = len(coefficient_vectors)
        size = len(PREFERENCE_DIMENSIONS)
        mean_vector = tuple(
            sum(vector[index] for vector in coefficient_vectors) / count
            for index in range(size)
        )

        covariance_rows: list[tuple[float, ...]] = []
        for row in range(size):
            values = []
            for column in range(size):
                aggregate = 0.0
                for posterior, coefficients in zip(
                    posteriors,
                    coefficient_vectors,
                    strict=True,
                ):
                    aggregate += posterior.covariance[row][column]
                    aggregate += (
                        (coefficients[row] - mean_vector[row])
                        * (coefficients[column] - mean_vector[column])
                    )
                value = aggregate / count
                if row == column:
                    value += self._regularization
                values.append(value)
            covariance_rows.append(tuple(values))

        return GaussianPreferencePrior(
            mean={
                dimension: mean_vector[index]
                for index, dimension in enumerate(PREFERENCE_DIMENSIONS)
            },
            covariance=tuple(covariance_rows),
            lower_bounds=posteriors[0].lower_bounds,
            upper_bounds=posteriors[0].upper_bounds,
            name=name,
            version=version,
            evidence_count=sum(posterior.evidence_count for posterior in posteriors),
            feature_schema_version=schema_version,
        )
