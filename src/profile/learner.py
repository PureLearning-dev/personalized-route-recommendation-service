"""根据路线成对选择学习长期基础偏好权重。

实现结合了两类论文思想：用具体路线选择反推可解释权重，以及使用群体先验
缓解冷启动。模型只学习画像，不生成路线，也不负责最终路线排序。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Sequence

from .exceptions import ProfileValidationError
from .models import (
    GroupPreferencePrior,
    LearningDiagnostics,
    PREFERENCE_DIMENSIONS,
    PairwisePreference,
    PreferenceDimension,
)
from .normalization import RouteAttributeNormalizer
from .optimization import project_to_probability_simplex, sigmoid, softplus


@dataclass(frozen=True, slots=True)
class WeightLearningConfig:
    """投影梯度下降的配置。

    学习率会按迭代次数的平方根衰减；四维问题通常能快速收敛。所有参数均
    开放注入，方便后续基于真实选择数据做离线标定。
    """

    max_iterations: int = 3000
    initial_learning_rate: float = 0.6
    tolerance: float = 1e-9
    prior_strength_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ProfileValidationError("最大迭代次数必须为正数")
        for field_name in ("initial_learning_rate", "tolerance", "prior_strength_multiplier"):
            value = float(getattr(self, field_name))
            if not isfinite(value) or value <= 0:
                raise ProfileValidationError(f"{field_name} 必须是有限的正数")
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True, slots=True)
class WeightLearningResult:
    """学习器输出，包含权重和诊断信息。"""

    weights: dict[PreferenceDimension, float]
    diagnostics: LearningDiagnostics


class PairwisePreferenceWeightLearner:
    """用带群体先验的成对 Logit 模型估计四项权重。

    对于用户选择的路线 ``chosen`` 和未选路线 ``rejected``，模型使用
    ``rejected - chosen`` 的归一化代价差。若某个权重与用户选择一致，该差值
    在相应维度上会提高所选路线的选择概率。
    """

    def __init__(
        self,
        normalizer: RouteAttributeNormalizer | None = None,
        config: WeightLearningConfig | None = None,
    ) -> None:
        self._normalizer = normalizer or RouteAttributeNormalizer()
        self._config = config or WeightLearningConfig()

    def fit(
        self,
        comparisons: Sequence[PairwisePreference],
        prior: GroupPreferencePrior,
    ) -> WeightLearningResult:
        """学习个人权重；没有个人数据时原样返回群体先验。"""

        if not comparisons:
            return WeightLearningResult(
                weights=dict(prior.weights),
                diagnostics=LearningDiagnostics(
                    converged=True,
                    iterations=0,
                    objective_value=0.0,
                    choice_consistency=0.0,
                    confidence=0.0,
                ),
            )

        # 预先把路线转换成固定顺序的差向量，避免每次迭代重复归一化。
        samples: list[tuple[list[float], float]] = []
        for comparison in comparisons:
            chosen = self._normalizer.normalize(comparison.chosen)
            rejected = self._normalizer.normalize(comparison.rejected)
            difference = [
                rejected[dimension] - chosen[dimension]
                for dimension in PREFERENCE_DIMENSIONS
            ]
            samples.append((difference, comparison.evidence_weight))

        total_evidence = sum(weight for _, weight in samples)
        prior_vector = [prior.weights[dimension] for dimension in PREFERENCE_DIMENSIONS]
        weights = list(prior_vector)

        # 等效样本量把群体先验解释为若干条虚拟比较；个人数据越多，先验占比越小。
        prior_strength = (
            prior.equivalent_sample_size * self._config.prior_strength_multiplier
        )
        denominator = total_evidence + prior_strength

        previous_objective = self._objective(
            weights, samples, prior_vector, prior_strength, denominator
        )
        converged = False
        iterations = 0

        for iteration in range(1, self._config.max_iterations + 1):
            gradient = [0.0] * len(PREFERENCE_DIMENSIONS)

            # 负对数似然的梯度：模型越不相信用户真实选择，修正幅度越大。
            for difference, sample_weight in samples:
                margin = sum(
                    weight * delta for weight, delta in zip(weights, difference, strict=True)
                )
                factor = sample_weight * (sigmoid(margin) - 1.0)
                for index, delta in enumerate(difference):
                    gradient[index] += factor * delta

            # L2 先验项把数据不足时的权重拉回群体中心，等价于可解释的冷启动保护。
            for index, current in enumerate(weights):
                gradient[index] += prior_strength * (current - prior_vector[index])
                gradient[index] /= denominator

            step_size = self._config.initial_learning_rate / sqrt(iteration)
            candidate = project_to_probability_simplex(
                [current - step_size * grad for current, grad in zip(weights, gradient, strict=True)]
            )
            objective = self._objective(
                candidate, samples, prior_vector, prior_strength, denominator
            )
            max_change = max(abs(new - old) for new, old in zip(candidate, weights, strict=True))

            weights = candidate
            iterations = iteration
            if max_change <= self._config.tolerance or abs(previous_objective - objective) <= self._config.tolerance:
                converged = True
                previous_objective = objective
                break
            previous_objective = objective

        # 一致性使用模型赋给已观察选择的平均概率，而不是简单的对/错计数。
        weighted_probability = 0.0
        for difference, sample_weight in samples:
            margin = sum(
                weight * delta for weight, delta in zip(weights, difference, strict=True)
            )
            weighted_probability += sample_weight * sigmoid(margin)
        choice_consistency = weighted_probability / total_evidence

        # 可信度同时考虑证据数量和选择一致性。它是工程诊断值，不是假装精确的统计置信区间。
        evidence_coverage = total_evidence / (total_evidence + prior.equivalent_sample_size)
        confidence = max(0.0, min(1.0, evidence_coverage * choice_consistency))

        learned = {
            dimension: weights[index]
            for index, dimension in enumerate(PREFERENCE_DIMENSIONS)
        }
        return WeightLearningResult(
            weights=learned,
            diagnostics=LearningDiagnostics(
                converged=converged,
                iterations=iterations,
                objective_value=previous_objective,
                choice_consistency=choice_consistency,
                confidence=confidence,
            ),
        )

    @staticmethod
    def _objective(
        weights: Sequence[float],
        samples: Sequence[tuple[list[float], float]],
        prior: Sequence[float],
        prior_strength: float,
        denominator: float,
    ) -> float:
        """计算平均负对数似然和先验惩罚，主要用于收敛判断与测试。"""

        data_loss = 0.0
        for difference, sample_weight in samples:
            margin = sum(
                weight * delta for weight, delta in zip(weights, difference, strict=True)
            )
            data_loss += sample_weight * softplus(-margin)
        prior_loss = 0.5 * prior_strength * sum(
            (current - center) ** 2 for current, center in zip(weights, prior, strict=True)
        )
        return (data_loss + prior_loss) / denominator

