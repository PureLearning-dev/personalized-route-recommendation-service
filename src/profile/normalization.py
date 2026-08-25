"""将路线的四项代价转换为统一尺度。"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from .models import (
    FeatureComparison,
    PREFERENCE_DIMENSIONS,
    PairwisePreference,
    PreferenceDimension,
    RouteAttributes,
    Vector,
)


# 固定尺度保证同一权重在不同候选路线集合中仍具有一致含义。
NORMALIZATION_SCALES: Mapping[PreferenceDimension, float] = MappingProxyType(
    {
        PreferenceDimension.TIME: 180.0,
        PreferenceDimension.COST: 100.0,
        PreferenceDimension.WALKING_DISTANCE: 3000.0,
        PreferenceDimension.TRANSFERS: 4.0,
    }
)


class NormalizedCostFeatureExtractor:
    """把一次路线选择转换成归一化四维比较数据。"""

    @staticmethod
    def _normalize(route: RouteAttributes) -> Vector:
        # 不对大于尺度的值截断，否则 200 分钟和 500 分钟会变成相同特征。
        return tuple(
            route.value_for(dimension) / NORMALIZATION_SCALES[dimension]
            for dimension in PREFERENCE_DIMENSIONS
        )

    def extract_comparison(self, preference: PairwisePreference) -> FeatureComparison:
        return FeatureComparison(
            chosen=self._normalize(preference.chosen),
            rejected=self._normalize(preference.rejected),
        )
