"""路线属性的固定尺度归一化。

偏好学习必须比较分钟、元、米和次数。若直接相加，数值较大的单位会无意中
获得更大影响；若每次都按当前候选集做 min-max，同一权重在不同出行中又会
失去一致含义。因此第一版使用可配置的固定业务尺度，并把极端值截断到 1。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .exceptions import ProfileValidationError
from .models import (
    FeatureComparison,
    PREFERENCE_DIMENSIONS,
    PairwisePreference,
    PreferenceDimension,
    RouteAttributes,
    RouteFeatureVector,
)


@dataclass(frozen=True, slots=True)
class NormalizationScale:
    """各项路线代价达到 1 时对应的业务尺度。

    默认值只是可运行的工程初值，不是论文给出的通用常数。实际部署时应根据
    服务区域的路线分布（例如 P90 或 P95）校准，并在模型版本内保持稳定。
    """

    time_minutes: float = 180.0
    cost: float = 100.0
    walking_distance_meters: float = 3000.0
    transfer_count: float = 4.0

    def __post_init__(self) -> None:
        for field_name in (
            "time_minutes",
            "cost",
            "walking_distance_meters",
            "transfer_count",
        ):
            value = float(getattr(self, field_name))
            if not isfinite(value) or value <= 0:
                raise ProfileValidationError(f"归一化尺度 {field_name} 必须是有限的正数")
            object.__setattr__(self, field_name, value)

    def for_dimension(self, dimension: PreferenceDimension) -> float:
        """把偏好维度映射到对应的固定尺度。"""

        return {
            PreferenceDimension.TIME: self.time_minutes,
            PreferenceDimension.COST: self.cost,
            PreferenceDimension.WALKING_DISTANCE: self.walking_distance_meters,
            PreferenceDimension.TRANSFERS: self.transfer_count,
        }[dimension]


class RouteAttributeNormalizer:
    """将原始路线属性转换为统一的 [0, 1] 代价向量。"""

    def __init__(self, scale: NormalizationScale | None = None) -> None:
        self._scale = scale or NormalizationScale()

    @property
    def scale(self) -> NormalizationScale:
        """暴露只读尺度，便于记录模型配置和编写可重复实验。"""

        return self._scale

    def normalize(self, route: RouteAttributes) -> dict[PreferenceDimension, float]:
        """归一化一条路线；数值越大表示该项代价越高。"""

        return {
            dimension: min(route.value_for(dimension) / self._scale.for_dimension(dimension), 1.0)
            for dimension in PREFERENCE_DIMENSIONS
        }


class NormalizedCostFeatureExtractor:
    """将业务路线转换为版本化的四维 FAVOUR 代价特征。"""

    def __init__(
        self,
        normalizer: RouteAttributeNormalizer | None = None,
        schema_version: str = "four-cost-v1",
    ) -> None:
        self._normalizer = normalizer or RouteAttributeNormalizer()
        self._schema_version = schema_version

    @property
    def schema_version(self) -> str:
        return self._schema_version

    def extract(self, route: RouteAttributes) -> RouteFeatureVector:
        return RouteFeatureVector(
            values=self._normalizer.normalize(route),
            schema_version=self._schema_version,
        )

    def extract_comparison(self, preference: PairwisePreference) -> FeatureComparison:
        return FeatureComparison(
            chosen=self.extract(preference.chosen),
            rejected=self.extract(preference.rejected),
            evidence_weight=preference.evidence_weight,
        )
