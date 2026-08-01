"""路线选择反推长期偏好权重所需的最小领域模型。

本模块只定义当前主流程真正使用的数据：四项偏好维度、路线属性、一次成对
选择、冷启动先验和学习诊断。数据结构与学习算法分离，今后增加新的画像维度
或替换学习器时，不需要把交互代码与数值计算混在一起修改。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isclose, isfinite
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
class LearningDiagnostics:
    """学习过程的诊断结果，不属于四项偏好权重本身。"""

    converged: bool
    iterations: int
    objective_value: float
    choice_consistency: float
    confidence: float
