"""按FAVOUR论文流程学习四维用户画像。"""

from __future__ import annotations

from collections.abc import Sequence

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
)
from .normalization import NormalizedCostFeatureExtractor


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
        lower_bounds={dimension: -20.0 for dimension in PREFERENCE_DIMENSIONS},
        upper_bounds={dimension: 0.0 for dimension in PREFERENCE_DIMENSIONS},
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
        mass_preference_prior: GaussianPreferenceModel | None = None,
    ) -> PreferenceLearningResult:
        """先获得MPP，再按公式（6）逐题形成个人画像。"""

        prior = mass_preference_prior or standard_mass_preference_prior()
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
