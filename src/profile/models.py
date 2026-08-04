"""FAVOUR 缩减特征实现使用的领域值对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import isclose, isfinite, sqrt
from typing import Mapping

from .exceptions import ProfileValidationError


class PreferenceDimension(StrEnum):
    """第一版长期偏好包含的四个可解释维度。"""

    TIME = "time"
    COST = "cost"
    WALKING_DISTANCE = "walking_distance"
    TRANSFERS = "transfers"


# 所有计算统一使用这个顺序，防止不同模块对同一组权重产生不同解释。
PREFERENCE_DIMENSIONS: tuple[PreferenceDimension, ...] = tuple(PreferenceDimension)
Matrix = tuple[tuple[float, ...], ...]


def _require_non_empty(value: str, field_name: str) -> str:
    """清理并校验必填字符串。"""

    cleaned = value.strip()
    if not cleaned:
        raise ProfileValidationError(f"{field_name} 不能为空")
    return cleaned


def _require_non_negative(value: float, field_name: str) -> float:
    """保证路线属性和权重是有限非负数。"""

    numeric = float(value)
    if not isfinite(numeric) or numeric < 0:
        raise ProfileValidationError(f"{field_name} 必须是有限的非负数")
    return numeric


def _copy_and_validate_weights(
    weights: Mapping[PreferenceDimension, float],
) -> dict[PreferenceDimension, float]:
    """复制并校验一组完整、非负且总和为1的四维权重。"""

    missing = set(PREFERENCE_DIMENSIONS) - set(weights)
    extra = set(weights) - set(PREFERENCE_DIMENSIONS)
    if missing or extra:
        raise ProfileValidationError(
            f"权重维度不完整，缺少={sorted(str(item) for item in missing)}，"
            f"多余={sorted(str(item) for item in extra)}"
        )

    copied = {
        dimension: _require_non_negative(weights[dimension], f"{dimension.value} 权重")
        for dimension in PREFERENCE_DIMENSIONS
    }
    if not isclose(sum(copied.values()), 1.0, abs_tol=1e-8):
        raise ProfileValidationError("画像权重之和必须为1")
    return copied


def _copy_finite_dimension_values(
    values: Mapping[PreferenceDimension, float],
    field_name: str,
    *,
    non_negative: bool = False,
) -> dict[PreferenceDimension, float]:
    """复制一组完整的四维有限数值。"""

    missing = set(PREFERENCE_DIMENSIONS) - set(values)
    extra = set(values) - set(PREFERENCE_DIMENSIONS)
    if missing or extra:
        raise ProfileValidationError(
            f"{field_name}维度不完整，缺少={sorted(str(item) for item in missing)}，"
            f"多余={sorted(str(item) for item in extra)}"
        )

    copied = {dimension: float(values[dimension]) for dimension in PREFERENCE_DIMENSIONS}
    for dimension, value in copied.items():
        if not isfinite(value) or (non_negative and value < 0):
            qualifier = "有限的非负数" if non_negative else "有限数"
            raise ProfileValidationError(f"{field_name}{dimension.value} 必须是{qualifier}")
    return copied


def _copy_and_validate_matrix(matrix: Matrix, field_name: str) -> Matrix:
    """校验与四维画像匹配的有限对称矩阵。"""

    size = len(PREFERENCE_DIMENSIONS)
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise ProfileValidationError(f"{field_name}必须是 {size}×{size} 矩阵")

    copied = tuple(tuple(float(value) for value in row) for row in matrix)
    for row in copied:
        if any(not isfinite(value) for value in row):
            raise ProfileValidationError(f"{field_name}必须只包含有限数")
    for row_index in range(size):
        if copied[row_index][row_index] <= 0:
            raise ProfileValidationError(f"{field_name}对角线必须为正数")
        for column_index in range(size):
            if not isclose(
                copied[row_index][column_index],
                copied[column_index][row_index],
                abs_tol=1e-10,
            ):
                raise ProfileValidationError(f"{field_name}必须是对称矩阵")
    return copied


@dataclass(frozen=True, slots=True)
class RouteAttributes:
    """一条比较路线中真正参与权重学习的四项代价属性。

    路线编号只用于区分两条路线；时间、费用、步行距离和换乘次数会经过归一化
    后进入学习器。路线生成和属性获取应由未来的上游路径服务负责。
    """

    route_id: str
    total_time_minutes: float
    total_cost: float
    walking_distance_meters: float
    transfer_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "route_id", _require_non_empty(self.route_id, "route_id"))
        object.__setattr__(
            self,
            "total_time_minutes",
            _require_non_negative(self.total_time_minutes, "总时间"),
        )
        object.__setattr__(
            self,
            "total_cost",
            _require_non_negative(self.total_cost, "总费用"),
        )
        object.__setattr__(
            self,
            "walking_distance_meters",
            _require_non_negative(self.walking_distance_meters, "步行距离"),
        )
        if (
            isinstance(self.transfer_count, bool)
            or not isinstance(self.transfer_count, int)
            or self.transfer_count < 0
        ):
            raise ProfileValidationError("换乘次数必须是非负整数")

    def value_for(self, dimension: PreferenceDimension) -> float:
        """按统一维度读取属性，供归一化器和学习器循环处理。"""

        return {
            PreferenceDimension.TIME: self.total_time_minutes,
            PreferenceDimension.COST: self.total_cost,
            PreferenceDimension.WALKING_DISTANCE: self.walking_distance_meters,
            PreferenceDimension.TRANSFERS: float(self.transfer_count),
        }[dimension]


@dataclass(frozen=True, slots=True)
class PairwisePreference:
    """一条“用户选择路线 chosen 而没有选择 rejected”的学习证据。

    ``evidence_weight`` 当前默认等于1。保留这个可选权重不会增加交互复杂度，
    但允许未来为不同质量的比较题设置不同强度，而无需修改学习器接口。
    """

    chosen: RouteAttributes
    rejected: RouteAttributes
    evidence_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.chosen.route_id == self.rejected.route_id:
            raise ProfileValidationError("成对比较中的两条路线必须不同")
        numeric_weight = float(self.evidence_weight)
        if not isfinite(numeric_weight) or numeric_weight <= 0:
            raise ProfileValidationError("证据权重必须是有限的正数")
        object.__setattr__(self, "evidence_weight", numeric_weight)


@dataclass(frozen=True, slots=True)
class RouteFeatureVector:
    """一条路线进入 FAVOUR 学习器的四维缩减特征。"""

    values: Mapping[PreferenceDimension, float]
    schema_version: str = "four-cost-v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            _copy_finite_dimension_values(
                self.values,
                "路线特征",
                non_negative=True,
            ),
        )
        object.__setattr__(self, "schema_version", _require_non_empty(self.schema_version, "特征版本"))

    def as_tuple(self) -> tuple[float, ...]:
        """按全局固定维度顺序返回数值向量。"""

        return tuple(self.values[dimension] for dimension in PREFERENCE_DIMENSIONS)


@dataclass(frozen=True, slots=True)
class FeatureComparison:
    """已选和未选路线的特征级成对证据。"""

    chosen: RouteFeatureVector
    rejected: RouteFeatureVector
    evidence_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.chosen.schema_version != self.rejected.schema_version:
            raise ProfileValidationError("成对路线的特征版本必须一致")
        numeric_weight = float(self.evidence_weight)
        if not isfinite(numeric_weight) or numeric_weight <= 0:
            raise ProfileValidationError("特征证据权重必须是有限的正数")
        object.__setattr__(self, "evidence_weight", numeric_weight)

    @property
    def schema_version(self) -> str:
        return self.chosen.schema_version

    def cost_difference(self) -> tuple[float, ...]:
        """返回 rejected - chosen，使正系数表示对代价更敏感。"""

        return tuple(
            rejected - chosen
            for chosen, rejected in zip(
                self.chosen.as_tuple(),
                self.rejected.as_tuple(),
                strict=True,
            )
        )


@dataclass(frozen=True, slots=True)
class GroupPreferencePrior:
    """个人选择较少时用于稳定结果的初始权重。

    ``equivalent_sample_size`` 控制初始权重相当于多少条比较证据。未来有真实
    群体统计后，可以注入新的权重；当前交互使用四项等权先验。
    """

    weights: Mapping[PreferenceDimension, float]
    equivalent_sample_size: float = 4.0
    name: str = "global"

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", _copy_and_validate_weights(self.weights))
        sample_size = float(self.equivalent_sample_size)
        if not isfinite(sample_size) or sample_size < 0:
            raise ProfileValidationError("群体先验的等效样本量必须是有限的非负数")
        object.__setattr__(self, "equivalent_sample_size", sample_size)
        object.__setattr__(self, "name", _require_non_empty(self.name, "群体先验名称"))

    @classmethod
    def uniform(cls, equivalent_sample_size: float = 4.0) -> GroupPreferencePrior:
        """创建四项等权的默认先验。"""

        equal_weight = 1.0 / len(PREFERENCE_DIMENSIONS)
        return cls(
            weights={dimension: equal_weight for dimension in PREFERENCE_DIMENSIONS},
            equivalent_sample_size=equivalent_sample_size,
            name="uniform",
        )


@dataclass(frozen=True, slots=True)
class GaussianPreferencePrior:
    """FAVOUR Gaussian 先验；系数采用正向代价敏感度表示。"""

    mean: Mapping[PreferenceDimension, float]
    covariance: Matrix
    lower_bounds: Mapping[PreferenceDimension, float]
    upper_bounds: Mapping[PreferenceDimension, float]
    name: str = "fixed-gaussian"
    version: str = "v1"
    evidence_count: int = 0
    feature_schema_version: str = "four-cost-v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mean",
            _copy_finite_dimension_values(self.mean, "先验均值", non_negative=True),
        )
        object.__setattr__(
            self,
            "covariance",
            _copy_and_validate_matrix(self.covariance, "先验协方差"),
        )
        lower = _copy_finite_dimension_values(
            self.lower_bounds,
            "系数下界",
            non_negative=True,
        )
        upper = _copy_finite_dimension_values(
            self.upper_bounds,
            "系数上界",
            non_negative=True,
        )
        for dimension in PREFERENCE_DIMENSIONS:
            if lower[dimension] > self.mean[dimension] or self.mean[dimension] > upper[dimension]:
                raise ProfileValidationError("先验均值必须处于系数边界内")
        object.__setattr__(self, "lower_bounds", lower)
        object.__setattr__(self, "upper_bounds", upper)
        object.__setattr__(self, "name", _require_non_empty(self.name, "先验名称"))
        object.__setattr__(self, "version", _require_non_empty(self.version, "先验版本"))
        object.__setattr__(
            self,
            "feature_schema_version",
            _require_non_empty(self.feature_schema_version, "特征版本"),
        )
        if isinstance(self.evidence_count, bool) or self.evidence_count < 0:
            raise ProfileValidationError("先验证据数量必须是非负整数")

    def mean_vector(self) -> tuple[float, ...]:
        return tuple(self.mean[dimension] for dimension in PREFERENCE_DIMENSIONS)

    def lower_bound_vector(self) -> tuple[float, ...]:
        return tuple(self.lower_bounds[dimension] for dimension in PREFERENCE_DIMENSIONS)

    def upper_bound_vector(self) -> tuple[float, ...]:
        return tuple(self.upper_bounds[dimension] for dimension in PREFERENCE_DIMENSIONS)


@dataclass(frozen=True, slots=True)
class PreferencePosterior:
    """Laplace 近似得到的个人偏好 Gaussian 后验。"""

    coefficients: Mapping[PreferenceDimension, float]
    covariance: Matrix
    lower_bounds: Mapping[PreferenceDimension, float]
    upper_bounds: Mapping[PreferenceDimension, float]
    evidence_count: int
    converged: bool
    iterations: int
    negative_log_posterior: float
    prior_name: str
    prior_version: str
    feature_schema_version: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coefficients",
            _copy_finite_dimension_values(
                self.coefficients,
                "后验系数",
                non_negative=True,
            ),
        )
        object.__setattr__(
            self,
            "covariance",
            _copy_and_validate_matrix(self.covariance, "后验协方差"),
        )
        object.__setattr__(
            self,
            "lower_bounds",
            _copy_finite_dimension_values(
                self.lower_bounds,
                "后验系数下界",
                non_negative=True,
            ),
        )
        object.__setattr__(
            self,
            "upper_bounds",
            _copy_finite_dimension_values(
                self.upper_bounds,
                "后验系数上界",
                non_negative=True,
            ),
        )
        for dimension in PREFERENCE_DIMENSIONS:
            if self.lower_bounds[dimension] > self.upper_bounds[dimension]:
                raise ProfileValidationError("后验系数下界不能大于上界")
            if not (
                self.lower_bounds[dimension]
                <= self.coefficients[dimension]
                <= self.upper_bounds[dimension]
            ):
                raise ProfileValidationError("后验系数必须处于边界内")
        if (
            isinstance(self.evidence_count, bool)
            or isinstance(self.iterations, bool)
            or self.evidence_count < 0
            or self.iterations < 0
        ):
            raise ProfileValidationError("后验证据数量和迭代次数必须为非负数")
        if not isfinite(float(self.negative_log_posterior)):
            raise ProfileValidationError("后验目标值必须是有限数")
        object.__setattr__(self, "prior_name", _require_non_empty(self.prior_name, "先验名称"))
        object.__setattr__(self, "prior_version", _require_non_empty(self.prior_version, "先验版本"))
        object.__setattr__(
            self,
            "feature_schema_version",
            _require_non_empty(self.feature_schema_version, "特征版本"),
        )

    def coefficient_vector(self) -> tuple[float, ...]:
        return tuple(self.coefficients[dimension] for dimension in PREFERENCE_DIMENSIONS)

    def as_incremental_prior(self) -> GaussianPreferencePrior:
        """把当前 Gaussian 后验作为下一条选择的先验。"""

        source_name = self.prior_name.removeprefix("posterior:")
        return GaussianPreferencePrior(
            mean=self.coefficients,
            covariance=self.covariance,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            name=f"posterior:{source_name}",
            version=self.prior_version,
            evidence_count=self.evidence_count,
            feature_schema_version=self.feature_schema_version,
        )


@dataclass(frozen=True, slots=True)
class LearningDiagnostics:
    """学习收敛、后验预测和不确定性诊断。"""

    converged: bool
    iterations: int
    objective_value: float
    choice_consistency: float
    posterior_standard_deviations: Mapping[PreferenceDimension, float] = field(
        default_factory=dict
    )
    evidence_count: int = 0

    def __post_init__(self) -> None:
        for field_name in ("objective_value", "choice_consistency"):
            if not isfinite(float(getattr(self, field_name))):
                raise ProfileValidationError(f"诊断字段 {field_name} 必须是有限数")
        if not 0.0 <= self.choice_consistency <= 1.0:
            raise ProfileValidationError("选择一致性必须位于 [0, 1]")
        if self.posterior_standard_deviations:
            object.__setattr__(
                self,
                "posterior_standard_deviations",
                _copy_finite_dimension_values(
                    self.posterior_standard_deviations,
                    "后验标准差",
                    non_negative=True,
                ),
            )
        if isinstance(self.evidence_count, bool) or self.evidence_count < 0:
            raise ProfileValidationError("诊断证据数量必须为非负数")


@dataclass(frozen=True, slots=True)
class PreferenceLearningResult:
    """学习结果同时保留模型系数、Gaussian 后验和展示权重。"""

    weights: dict[PreferenceDimension, float]
    coefficients: dict[PreferenceDimension, float]
    posterior: PreferencePosterior
    diagnostics: LearningDiagnostics

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", _copy_and_validate_weights(self.weights))
        coefficients = _copy_finite_dimension_values(
            self.coefficients,
            "模型系数",
            non_negative=True,
        )
        if any(
            not isclose(
                coefficients[dimension],
                self.posterior.coefficients[dimension],
                abs_tol=1e-10,
            )
            for dimension in PREFERENCE_DIMENSIONS
        ):
            raise ProfileValidationError("结果系数必须与后验系数一致")
        object.__setattr__(self, "coefficients", coefficients)
