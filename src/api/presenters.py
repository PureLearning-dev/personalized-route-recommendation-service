"""应用记录到 API 响应的转换。"""

from __future__ import annotations

from ..application.records import ProfileRecord
from ..profile.models import PreferenceDimension
from .schemas.profiles import ProfileData, ProfileResponse


def present_profile(profile: ProfileRecord) -> ProfileResponse:
    weights = {
        dimension.value: profile.result.weights[dimension] for dimension in PreferenceDimension
    }
    return ProfileResponse(
        user_id=profile.user_id,
        profile=ProfileData(
            evidence_count=profile.evidence_count,
            converged=profile.converged,
            coefficients={
                dimension.value: profile.result.utility_coefficients[dimension]
                for dimension in PreferenceDimension
            },
            weights=weights,
            percentages={key: value * 100.0 for key, value in weights.items()},
            covariance=[list(row) for row in profile.result.posterior.covariance],
            standard_deviations={
                dimension.value: value
                for dimension, value in profile.result.standard_deviations.items()
            },
            created_at=profile.created_at,
            updated_at=profile.updated_at,
        ),
    )
