from __future__ import annotations

from .common import ApiModel
from .profiles import ProfileResponse
from .recommendations import RouteInputSchema


class ChoiceCreate(ApiModel):
    chosen_route: RouteInputSchema
    rejected_route: RouteInputSchema


class ChoiceLearningResponse(ApiModel):
    learning_applied: bool
    profile: ProfileResponse
