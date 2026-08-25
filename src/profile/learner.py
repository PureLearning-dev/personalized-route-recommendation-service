"""按FAVOUR论文流程学习四维用户画像。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite

from .exceptions import ProfileValidationError
from .inference import (
    FavourLaplaceInference,
    FavourPosteriorPredictor,
    MassPreferencePriorEstimator,
)
from .models import (
    FeatureComparison,
    GaussianPreferenceModel,
    Matrix,
    PREFERENCE_DIMENSIONS,
    PairwisePreference,
    PreferenceLearningResult,
    PreferenceDimension,
    PreferencePreset,
)
from .normalization import NormalizedCostFeatureExtractor


_COEFFICIENT_LOWER_BOUND = -20.0
_COEFFICIENT_UPPER_BOUND = 0.0
_DEFAULT_PRESET_STRENGTH = 4.0
_DEFAULT_PRESET_VARIANCE = 1.0

_PRESET_PREFERENCE_WEIGHTS: Mapping[
    PreferencePreset,
    Mapping[PreferenceDimension, float],
] = {
    PreferencePreset.BALANCED: {
        dimension: 0.25 for dimension in PREFERENCE_DIMENSIONS
    },
    PreferencePreset.TIME_PRIORITY: {
        PreferenceDimension.TIME: 0.70,
        PreferenceDimension.COST: 0.10,
        PreferenceDimension.WALKING_DISTANCE: 0.10,
        PreferenceDimension.TRANSFERS: 0.10,
    },
    PreferencePreset.COST_PRIORITY: {
        PreferenceDimension.TIME: 0.10,
        PreferenceDimension.COST: 0.70,
        PreferenceDimension.WALKING_DISTANCE: 0.10,
        PreferenceDimension.TRANSFERS: 0.10,
    },
    PreferencePreset.LOW_WALKING: {
        PreferenceDimension.TIME: 0.10,
        PreferenceDimension.COST: 0.10,
        PreferenceDimension.WALKING_DISTANCE: 0.70,
        PreferenceDimension.TRANSFERS: 0.10,
    },
    PreferencePreset.LOW_TRANSFERS: {
        PreferenceDimension.TIME: 0.10,
        PreferenceDimension.COST: 0.10,
        PreferenceDimension.WALKING_DISTANCE: 0.10,
        PreferenceDimension.TRANSFERS: 0.70,
    },
}


def preset_preference_weights(
    preset: PreferencePreset | str,
) -> dict[PreferenceDimension, float]:
    """返回一个预设画像的四维权重副本，避免调用方修改全局配置。"""

    try:
        resolved = PreferencePreset(preset)
    except (TypeError, ValueError) as error:
        available = "、".join(item.value for item in PreferencePreset)
        raise ProfileValidationError(
            f"未知预设画像 {preset!r}，可选值：{available}"
        ) from error
    return dict(_PRESET_PREFERENCE_WEIGHTS[resolved])


def preference_prior_from_weights(
    weights: Mapping[PreferenceDimension, float],
    *,
    coefficient_strength: float = _DEFAULT_PRESET_STRENGTH,
    variance: float = _DEFAULT_PRESET_VARIANCE,
) -> GaussianPreferenceModel:
    """把易读的正权重转换成可参与贝叶斯更新的Gaussian代价系数先验。

    输入权重会自动归一化，因此既可传入 ``0.7/0.1``，也可传入
    ``70/10``。画像系数必须为非正数，所以归一化权重会乘以负的总体强度；
    权重比例决定初始偏好方向，方差决定后续选择证据改写预设的难易程度。
    """

    if set(weights) != set(PREFERENCE_DIMENSIONS):
        raise ProfileValidationError("预设画像权重必须包含完整的四个画像维度")

    copied = {
        dimension: float(weights[dimension])
        for dimension in PREFERENCE_DIMENSIONS
    }
    if any(not isfinite(value) or value < 0.0 for value in copied.values()):
        raise ProfileValidationError("预设画像权重必须是有限的非负数")
    total = sum(copied.values())
    if total <= 0.0:
        raise ProfileValidationError("预设画像权重之和必须大于0")

    strength = float(coefficient_strength)
    if not isfinite(strength) or not 0.0 < strength <= abs(_COEFFICIENT_LOWER_BOUND):
        raise ProfileValidationError("预设画像系数强度必须位于 (0, 20] 区间")
    prior_variance = float(variance)
    if not isfinite(prior_variance) or prior_variance <= 0.0:
        raise ProfileValidationError("预设画像方差必须是有限的正数")

    size = len(PREFERENCE_DIMENSIONS)
    covariance: Matrix = tuple(
        tuple(
            prior_variance if row == column else 0.0
            for column in range(size)
        )
        for row in range(size)
    )
    return GaussianPreferenceModel(
        mean={
            dimension: -strength * copied[dimension] / total
            for dimension in PREFERENCE_DIMENSIONS
        },
        covariance=covariance,
        lower_bounds={
            dimension: _COEFFICIENT_LOWER_BOUND
            for dimension in PREFERENCE_DIMENSIONS
        },
        upper_bounds={
            dimension: _COEFFICIENT_UPPER_BOUND
            for dimension in PREFERENCE_DIMENSIONS
        },
    )


def preset_preference_prior(
    preset: PreferencePreset | str,
    *,
    coefficient_strength: float = _DEFAULT_PRESET_STRENGTH,
    variance: float = _DEFAULT_PRESET_VARIANCE,
) -> GaussianPreferenceModel:
    """将命名预设画像转换为个人学习流程可以直接使用的先验。"""

    return preference_prior_from_weights(
        preset_preference_weights(preset),
        coefficient_strength=coefficient_strength,
        variance=variance,
    )


def standard_mass_preference_prior() -> GaussianPreferenceModel:
    """论文图2的MPP初值 N(0, I)，并限制四项代价系数不大于0。"""

    size = len(PREFERENCE_DIMENSIONS)
    identity: Matrix = tuple(
        tuple(1.0 if row == column else 0.0 for column in range(size))
        for row in range(size)
    )
    return GaussianPreferenceModel(
        mean={dimension: 0.0 for dimension in PREFERENCE_DIMENSIONS},
        covariance=identity,
        lower_bounds={
            dimension: _COEFFICIENT_LOWER_BOUND
            for dimension in PREFERENCE_DIMENSIONS
        },
        upper_bounds={
            dimension: _COEFFICIENT_UPPER_BOUND
            for dimension in PREFERENCE_DIMENSIONS
        },
    )


class PairwisePreferenceWeightLearner:
    """FAVOUR的MPP、逐题更新、Laplace后验和预测统一入口。"""

    def __init__(self) -> None:
        self._extractor = NormalizedCostFeatureExtractor()
        self._inference = FavourLaplaceInference()
        self._mpp_estimator = MassPreferencePriorEstimator(self._inference)
        self._predictor = FavourPosteriorPredictor()

    def _extract(
        self,
        comparisons: Sequence[PairwisePreference],
    ) -> tuple[FeatureComparison, ...]:
        return tuple(
            self._extractor.extract_comparison(comparison)
            for comparison in comparisons
        )

    def fit(
        self,
        comparisons: Sequence[PairwisePreference],
        group_histories: Sequence[Sequence[PairwisePreference]] = (),
        *,
        preference_preset: PreferencePreset | str | None = None,
    ) -> PreferenceLearningResult:
        """先确定预设或群体MPP，再按公式（6）逐题形成个人画像。"""

        if preference_preset is not None:
            prior = preset_preference_prior(preference_preset)
        else:
            prior = standard_mass_preference_prior()
        if group_histories:
            prior = self._mpp_estimator.refine(
                tuple(self._extract(history) for history in group_histories),
                prior,
            )

        observations = self._extract(comparisons)
        posterior, converged = self._inference.update_incrementally(
            observations,
            prior,
        )
        probabilities = tuple(
            self._predictor.probability(posterior, observation)
            for observation in observations
        )
        sensitivities = {
            dimension: max(0.0, -posterior.mean[dimension])
            for dimension in PREFERENCE_DIMENSIONS
        }
        total = sum(sensitivities.values())
        if total <= 1e-12:
            equal = 1.0 / len(PREFERENCE_DIMENSIONS)
            weights = {dimension: equal for dimension in PREFERENCE_DIMENSIONS}
        else:
            weights = {
                dimension: sensitivities[dimension] / total
                for dimension in PREFERENCE_DIMENSIONS
            }
        return PreferenceLearningResult(
            posterior=posterior,
            weights=weights,
            evidence_count=len(observations),
            converged=converged,
            choice_probabilities=probabilities,
        )
