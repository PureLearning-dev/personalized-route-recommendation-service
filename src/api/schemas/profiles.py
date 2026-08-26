from __future__ import annotations

from datetime import datetime
from uuid import UUID

from .common import ApiModel


class ProfileData(ApiModel):
    evidence_count: int
    converged: bool
    coefficients: dict[str, float]
    weights: dict[str, float]
    percentages: dict[str, float]
    covariance: list[list[float]]
    standard_deviations: dict[str, float]
    created_at: datetime
    updated_at: datetime


class ProfileResponse(ApiModel):
    user_id: UUID
    profile: ProfileData
