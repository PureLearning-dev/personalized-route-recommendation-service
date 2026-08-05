"""将路线的四项代价转换为统一尺度。"""

from __future__ import annotations

from .models import (
    FeatureComparison,
    PREFERENCE_DIMENSIONS,
    PairwisePreference,
    PreferenceDimension,
    RouteAttributes,
    Vector,
)


NORMALIZATION_SCALES = {
    PreferenceDimension.TIME: 180.0,
    PreferenceDimension.COST: 100.0,
    PreferenceDimension.WALKING_DISTANCE: 3000.0,
    PreferenceDimension.TRANSFERS: 4.0,
}


class NormalizedCostFeatureExtractor:
    """把一次路线选择转换成归一化四维比较数据。"""

    @staticmethod
    def _normalize(route: RouteAttributes) -> Vector:
        return tuple(
            min(
                route.value_for(dimension) / NORMALIZATION_SCALES[dimension],
                1.0,
            )
            for dimension in PREFERENCE_DIMENSIONS
        )

    def extract_comparison(self, preference: PairwisePreference) -> FeatureComparison:
        return FeatureComparison(
            chosen=self._normalize(preference.chosen),
            rejected=self._normalize(preference.rejected),
        )
