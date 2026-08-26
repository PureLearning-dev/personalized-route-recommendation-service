"""应用层与持久化层之间使用的轻量记录。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from ..profile.models import (
    PREFERENCE_DIMENSIONS,
    GaussianPreferenceModel,
    PreferenceDimension,
    PreferenceLearningResult,
)


@dataclass(frozen=True, slots=True)
class UserRecord:
    id: UUID
    external_user_id: str
    initialization_mode: str
    preset_name: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ProfileRecord:
    user_id: UUID
    result: PreferenceLearningResult
    created_at: datetime
    updated_at: datetime

    @property
    def evidence_count(self) -> int:
        return self.result.evidence_count

    @property
    def converged(self) -> bool:
        return self.result.converged


@dataclass(frozen=True, slots=True)
class RecommendationResultRecord:
    profile: ProfileRecord
    ranking_mode: str
    candidate_count: int
    feasible_count: int
    ranked_routes: tuple[dict[str, Any], ...]
    rejected_routes: tuple[dict[str, Any], ...]
    explanation: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ChoiceLearningRecord:
    profile: ProfileRecord
    learning_applied: bool


def restore_learning_result(
    *,
    coefficients: dict[str, float],
    covariance: tuple[tuple[float, ...], ...],
    evidence_count: int,
    converged: bool,
) -> PreferenceLearningResult:
    """从当前画像表恢复可继续学习的 Gaussian 后验。"""

    means = {PreferenceDimension(key): value for key, value in coefficients.items()}
    sensitivities = {dimension: max(0.0, -means[dimension]) for dimension in PREFERENCE_DIMENSIONS}
    total = sum(sensitivities.values())
    if total <= 1e-12:
        weights = {
            dimension: 1.0 / len(PREFERENCE_DIMENSIONS) for dimension in PREFERENCE_DIMENSIONS
        }
    else:
        weights = {
            dimension: sensitivities[dimension] / total for dimension in PREFERENCE_DIMENSIONS
        }
    return PreferenceLearningResult(
        posterior=GaussianPreferenceModel(
            mean=means,
            covariance=covariance,
            lower_bounds={dimension: -20.0 for dimension in PREFERENCE_DIMENSIONS},
            upper_bounds={dimension: 0.0 for dimension in PREFERENCE_DIMENSIONS},
        ),
        weights=weights,
        evidence_count=evidence_count,
        converged=converged,
        choice_probabilities=(),
    )
