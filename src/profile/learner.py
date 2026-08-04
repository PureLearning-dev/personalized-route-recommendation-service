"""FAVOUR 四维缩减模型的简洁学习入口。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

from .exceptions import ProfileValidationError
from .inference import (
    BradleyTerryLogitLikelihood,
    FavourLaplacePosteriorEstimator,
    FavourPosteriorPredictor,
)
from .models import (
    GaussianPreferencePrior,
    GroupPreferencePrior,
    PairwisePreference,
    PreferenceLearningResult,
)
from .normalization import NormalizedCostFeatureExtractor, RouteAttributeNormalizer
from .optimization import BoxConstrainedNewtonOptimizer, NewtonOptimizerConfig
from .priors import FixedGaussianPriorProvider, LegacyGroupPriorAdapter
from .service import (
    FavourPreferenceLearningService,
    PosteriorRepository,
    RelativeWeightProfilePresenter,
)


@dataclass(frozen=True, slots=True)
class WeightLearningConfig:
    """FAVOUR 后验众数优化与旧先验转换配置。"""

    max_iterations: int = 200
    tolerance: float = 1e-9
    initial_jitter: float = 1e-10
    line_search_shrink: float = 0.5
    armijo_constant: float = 1e-4
    minimum_step_size: float = 1e-10
    coefficient_scale: float = 4.0
    prior_variance_scale: float = 4.0
    coefficient_upper_bound: float = 20.0
    prior_strength_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.max_iterations, bool) or self.max_iterations <= 0:
            raise ProfileValidationError("最大迭代次数必须为正整数")
        if not 0 < self.line_search_shrink < 1:
            raise ProfileValidationError("line_search_shrink 必须位于 (0, 1)")
        if not 0 < self.armijo_constant < 1:
            raise ProfileValidationError("armijo_constant 必须位于 (0, 1)")
        for field_name in (
            "tolerance",
            "initial_jitter",
            "minimum_step_size",
            "coefficient_scale",
            "prior_variance_scale",
            "coefficient_upper_bound",
            "prior_strength_multiplier",
        ):
            value = float(getattr(self, field_name))
            if not isfinite(value) or value <= 0:
                raise ProfileValidationError(f"{field_name} 必须是有限的正数")
            object.__setattr__(self, field_name, value)


class PairwisePreferenceWeightLearner:
    """通过路线成对选择学习四维 FAVOUR 偏好后验。

    内部系数是非负、无需和为 1 的代价敏感度；``result.weights`` 只是展示层
    归一化后的四维画像百分比，``result.coefficients`` 才是模型实际使用的系数。
    """

    def __init__(
        self,
        normalizer: RouteAttributeNormalizer | None = None,
        config: WeightLearningConfig | None = None,
    ) -> None:
        self._config = config or WeightLearningConfig()
        self._feature_extractor = NormalizedCostFeatureExtractor(normalizer)
        likelihood = BradleyTerryLogitLikelihood()
        optimizer = BoxConstrainedNewtonOptimizer(
            NewtonOptimizerConfig(
                max_iterations=self._config.max_iterations,
                tolerance=self._config.tolerance,
                initial_jitter=self._config.initial_jitter,
                line_search_shrink=self._config.line_search_shrink,
                armijo_constant=self._config.armijo_constant,
                minimum_step_size=self._config.minimum_step_size,
            )
        )
        self._estimator = FavourLaplacePosteriorEstimator(optimizer, likelihood)
        self._predictor = FavourPosteriorPredictor()
        self._presenter = RelativeWeightProfilePresenter()
        self._legacy_prior_adapter = LegacyGroupPriorAdapter(
            coefficient_scale=self._config.coefficient_scale,
            base_variance=(
                self._config.prior_variance_scale
                / self._config.prior_strength_multiplier
            ),
            upper_bound=self._config.coefficient_upper_bound,
            feature_schema_version=self._feature_extractor.schema_version,
        )

    def _convert_prior(
        self,
        prior: GroupPreferencePrior | GaussianPreferencePrior,
    ) -> GaussianPreferencePrior:
        if isinstance(prior, GaussianPreferencePrior):
            return prior
        if isinstance(prior, GroupPreferencePrior):
            return self._legacy_prior_adapter.convert(prior)
        raise ProfileValidationError(
            "先验必须是 GroupPreferencePrior 或 GaussianPreferencePrior"
        )

    def create_service(
        self,
        prior: GroupPreferencePrior | GaussianPreferencePrior,
        repository: PosteriorRepository | None = None,
    ) -> FavourPreferenceLearningService:
        """创建可执行批量学习、增量更新和选择预测的应用服务。"""

        gaussian_prior = self._convert_prior(prior)
        return FavourPreferenceLearningService(
            feature_extractor=self._feature_extractor,
            prior_provider=FixedGaussianPriorProvider(gaussian_prior),
            estimator=self._estimator,
            predictor=self._predictor,
            presenter=self._presenter,
            repository=repository,
        )

    def fit(
        self,
        comparisons: Sequence[PairwisePreference],
        prior: GroupPreferencePrior | GaussianPreferencePrior,
    ) -> PreferenceLearningResult:
        """从一批路线选择学习后验，并返回模型系数及四维展示画像。"""

        return self.create_service(prior).fit(comparisons)
