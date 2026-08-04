"""编排特征、先验、推断、预测、存储与展示的应用服务。"""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from typing import Protocol

from .exceptions import ProfileStateError, ProfileValidationError
from .models import (
    FeatureComparison,
    GaussianPreferencePrior,
    LearningDiagnostics,
    PREFERENCE_DIMENSIONS,
    PairwisePreference,
    PreferenceDimension,
    PreferenceLearningResult,
    PreferencePosterior,
)


class RouteFeatureExtractor(Protocol):
    def extract_comparison(self, preference: PairwisePreference) -> FeatureComparison: ...


class PreferencePriorProvider(Protocol):
    def get_prior(self) -> GaussianPreferencePrior: ...


class PosteriorEstimator(Protocol):
    def fit(
        self,
        observations: Sequence[FeatureComparison],
        prior: GaussianPreferencePrior,
    ) -> PreferencePosterior: ...

    def update(
        self,
        posterior: PreferencePosterior,
        observation: FeatureComparison,
    ) -> PreferencePosterior: ...


class PreferencePredictor(Protocol):
    def probability_chosen(
        self,
        posterior: PreferencePosterior,
        observation: FeatureComparison,
    ) -> float: ...


class PosteriorRepository(Protocol):
    def load(self, user_id: str) -> PreferencePosterior | None: ...

    def save(self, user_id: str, posterior: PreferencePosterior) -> None: ...


class ProfilePresenter(Protocol):
    def relative_weights(
        self,
        posterior: PreferencePosterior,
    ) -> dict[PreferenceDimension, float]: ...

    def standard_deviations(
        self,
        posterior: PreferencePosterior,
    ) -> dict[PreferenceDimension, float]: ...


class RelativeWeightProfilePresenter:
    """把后验转换为四维展示百分比和标准差。"""

    def relative_weights(
        self,
        posterior: PreferencePosterior,
    ) -> dict[PreferenceDimension, float]:
        total = sum(posterior.coefficients.values())
        if total <= 1e-12:
            equal = 1.0 / len(PREFERENCE_DIMENSIONS)
            return {dimension: equal for dimension in PREFERENCE_DIMENSIONS}
        return {
            dimension: posterior.coefficients[dimension] / total
            for dimension in PREFERENCE_DIMENSIONS
        }

    def standard_deviations(
        self,
        posterior: PreferencePosterior,
    ) -> dict[PreferenceDimension, float]:
        return {
            dimension: sqrt(max(0.0, posterior.covariance[index][index]))
            for index, dimension in enumerate(PREFERENCE_DIMENSIONS)
        }


class InMemoryPosteriorRepository:
    """供演示和测试使用的内存后验仓库。"""

    def __init__(self) -> None:
        self._posteriors: dict[str, PreferencePosterior] = {}

    @staticmethod
    def _validate_user_id(user_id: str) -> str:
        cleaned = user_id.strip()
        if not cleaned:
            raise ProfileValidationError("user_id 不能为空")
        return cleaned

    def load(self, user_id: str) -> PreferencePosterior | None:
        return self._posteriors.get(self._validate_user_id(user_id))

    def save(self, user_id: str, posterior: PreferencePosterior) -> None:
        self._posteriors[self._validate_user_id(user_id)] = posterior


class FavourPreferenceLearningService:
    """只负责 FAVOUR 学习用例的流程编排，不包含具体数学或存储实现。"""

    def __init__(
        self,
        feature_extractor: RouteFeatureExtractor,
        prior_provider: PreferencePriorProvider,
        estimator: PosteriorEstimator,
        predictor: PreferencePredictor,
        presenter: ProfilePresenter,
        repository: PosteriorRepository | None = None,
    ) -> None:
        self._feature_extractor = feature_extractor
        self._prior_provider = prior_provider
        self._estimator = estimator
        self._predictor = predictor
        self._presenter = presenter
        self._repository = repository

    def _extract(
        self,
        comparisons: Sequence[PairwisePreference],
    ) -> tuple[FeatureComparison, ...]:
        return tuple(
            self._feature_extractor.extract_comparison(comparison)
            for comparison in comparisons
        )

    def _result(
        self,
        posterior: PreferencePosterior,
        observations: Sequence[FeatureComparison],
    ) -> PreferenceLearningResult:
        total_weight = sum(observation.evidence_weight for observation in observations)
        if total_weight:
            mean_probability = sum(
                observation.evidence_weight
                * self._predictor.probability_chosen(posterior, observation)
                for observation in observations
            ) / total_weight
        else:
            mean_probability = 0.0

        diagnostics = LearningDiagnostics(
            converged=posterior.converged,
            iterations=posterior.iterations,
            objective_value=posterior.negative_log_posterior,
            choice_consistency=mean_probability,
            posterior_standard_deviations=self._presenter.standard_deviations(posterior),
            evidence_count=posterior.evidence_count,
        )
        return PreferenceLearningResult(
            weights=self._presenter.relative_weights(posterior),
            coefficients=dict(posterior.coefficients),
            posterior=posterior,
            diagnostics=diagnostics,
        )

    def fit(
        self,
        comparisons: Sequence[PairwisePreference],
        prior: GaussianPreferencePrior | None = None,
    ) -> PreferenceLearningResult:
        observations = self._extract(comparisons)
        posterior = self._estimator.fit(
            observations,
            prior or self._prior_provider.get_prior(),
        )
        return self._result(posterior, observations)

    def update_user(
        self,
        user_id: str,
        comparison: PairwisePreference,
    ) -> PreferenceLearningResult:
        if self._repository is None:
            raise ProfileStateError("增量更新需要注入 PosteriorRepository")
        observation = self._feature_extractor.extract_comparison(comparison)
        previous = self._repository.load(user_id)
        if previous is None:
            posterior = self._estimator.fit(
                (observation,),
                self._prior_provider.get_prior(),
            )
        else:
            posterior = self._estimator.update(previous, observation)
        self._repository.save(user_id, posterior)
        return self._result(posterior, (observation,))

    def predict_user_choice(
        self,
        user_id: str,
        comparison: PairwisePreference,
    ) -> float:
        if self._repository is None:
            raise ProfileStateError("用户预测需要注入 PosteriorRepository")
        posterior = self._repository.load(user_id)
        if posterior is None:
            raise ProfileStateError(f"用户 {user_id!r} 尚无后验画像")
        observation = self._feature_extractor.extract_comparison(comparison)
        return self._predictor.probability_chosen(posterior, observation)
