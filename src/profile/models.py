"""FAVOUR 四维缩减模型的数据对象。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isclose, isfinite, sqrt
from typing import Mapping

from .exceptions import ProfileValidationError


class PreferenceDimension(StrEnum):
    TIME = "time"
    COST = "cost"
    WALKING_DISTANCE = "walking_distance"
    TRANSFERS = "transfers"


PREFERENCE_DIMENSIONS: tuple[PreferenceDimension, ...] = tuple(PreferenceDimension)
Vector = tuple[float, ...]
Matrix = tuple[tuple[float, ...], ...]


def _dimension_values(
    values: Mapping[PreferenceDimension, float],
    field_name: str,
) -> dict[PreferenceDimension, float]:
    if set(values) != set(PREFERENCE_DIMENSIONS):
        raise ProfileValidationError(f"{field_name}必须包含完整的四个画像维度")
    copied = {dimension: float(values[dimension]) for dimension in PREFERENCE_DIMENSIONS}
    if any(not isfinite(value) for value in copied.values()):
        raise ProfileValidationError(f"{field_name}必须是有限数")
    return copied


def _matrix(matrix: Matrix, field_name: str) -> Matrix:
    size = len(PREFERENCE_DIMENSIONS)
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise ProfileValidationError(f"{field_name}必须是 {size}×{size} 矩阵")
    copied = tuple(tuple(float(value) for value in row) for row in matrix)
    if any(not isfinite(value) for row in copied for value in row):
        raise ProfileValidationError(f"{field_name}必须只包含有限数")
    for row in range(size):
        if copied[row][row] <= 0:
            raise ProfileValidationError(f"{field_name}对角线必须为正数")
        for column in range(size):
            if not isclose(copied[row][column], copied[column][row], abs_tol=1e-10):
                raise ProfileValidationError(f"{field_name}必须是对称矩阵")
    return copied


@dataclass(frozen=True, slots=True)
class RouteAttributes:
    """一条路线的四项原始代价属性。"""

    route_id: str
    total_time_minutes: float
    total_cost: float
    walking_distance_meters: float
    transfer_count: int

    def __post_init__(self) -> None:
        if not self.route_id.strip():
            raise ProfileValidationError("route_id 不能为空")
        for field_name in (
            "total_time_minutes",
            "total_cost",
            "walking_distance_meters",
        ):
            value = float(getattr(self, field_name))
            if not isfinite(value) or value < 0:
                raise ProfileValidationError(f"{field_name} 必须是有限的非负数")
            object.__setattr__(self, field_name, value)
        if (
            isinstance(self.transfer_count, bool)
            or not isinstance(self.transfer_count, int)
            or self.transfer_count < 0
        ):
            raise ProfileValidationError("transfer_count 必须是非负整数")

    def value_for(self, dimension: PreferenceDimension) -> float:
        return {
            PreferenceDimension.TIME: self.total_time_minutes,
            PreferenceDimension.COST: self.total_cost,
            PreferenceDimension.WALKING_DISTANCE: self.walking_distance_meters,
            PreferenceDimension.TRANSFERS: float(self.transfer_count),
        }[dimension]


@dataclass(frozen=True, slots=True)
class PairwisePreference:
    """一次“选择 chosen、拒绝 rejected”的路线比较。"""

    chosen: RouteAttributes
    rejected: RouteAttributes

    def __post_init__(self) -> None:
        if self.chosen.route_id == self.rejected.route_id:
            raise ProfileValidationError("成对比较中的两条路线必须不同")


@dataclass(frozen=True, slots=True)
class FeatureComparison:
    """论文训练对 (r_t, q_t) 的四维特征。"""

    chosen: Vector
    rejected: Vector

    def utility_difference(self) -> Vector:
        """返回论文公式中的 u(r_t) - u(q_t)。"""

        return tuple(
            chosen - rejected
            for chosen, rejected in zip(self.chosen, self.rejected, strict=True)
        )


@dataclass(frozen=True, slots=True)
class GaussianPreferenceModel:
    """论文中的 Gaussian MPP、先验或 Laplace 后验 N(mean, covariance)。"""

    mean: Mapping[PreferenceDimension, float]
    covariance: Matrix
    lower_bounds: Mapping[PreferenceDimension, float]
    upper_bounds: Mapping[PreferenceDimension, float]

    def __post_init__(self) -> None:
        mean = _dimension_values(self.mean, "Gaussian均值")
        lower = _dimension_values(self.lower_bounds, "系数下界")
        upper = _dimension_values(self.upper_bounds, "系数上界")
        if any(not lower[d] <= mean[d] <= upper[d] for d in PREFERENCE_DIMENSIONS):
            raise ProfileValidationError("Gaussian均值必须位于系数边界内")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", _matrix(self.covariance, "Gaussian协方差"))
        object.__setattr__(self, "lower_bounds", lower)
        object.__setattr__(self, "upper_bounds", upper)

    def mean_vector(self) -> Vector:
        return tuple(self.mean[dimension] for dimension in PREFERENCE_DIMENSIONS)

    def lower_bound_vector(self) -> Vector:
        return tuple(self.lower_bounds[dimension] for dimension in PREFERENCE_DIMENSIONS)

    def upper_bound_vector(self) -> Vector:
        return tuple(self.upper_bounds[dimension] for dimension in PREFERENCE_DIMENSIONS)


@dataclass(frozen=True, slots=True)
class PreferenceLearningResult:
    """论文画像后验及便于读取的四维相对敏感度。"""

    posterior: GaussianPreferenceModel
    weights: Mapping[PreferenceDimension, float]
    evidence_count: int
    converged: bool
    choice_probabilities: tuple[float, ...]

    @property
    def utility_coefficients(self) -> Mapping[PreferenceDimension, float]:
        """论文效用函数实际使用的系数 w；代价维度应为非正数。"""

        return self.posterior.mean

    @property
    def standard_deviations(self) -> dict[PreferenceDimension, float]:
        return {
            dimension: sqrt(self.posterior.covariance[index][index])
            for index, dimension in enumerate(PREFERENCE_DIMENSIONS)
        }
