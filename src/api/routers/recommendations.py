from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter

from ...application.commands import (
    ChoiceCommand,
    ConstraintsInput,
    JndInput,
    RecommendationCommand,
    RouteInput,
)
from ...application.services import FeedbackService, RecommendationService
from ..dependencies import Repository, SettingsDependency
from ..presenters import present_profile
from ..schemas.feedback import ChoiceCreate, ChoiceLearningResponse
from ..schemas.recommendations import (
    RecommendationCreate,
    RecommendationResponse,
    RouteInputSchema,
)

router = APIRouter(tags=["recommendations"])


def _route_input(route: RouteInputSchema) -> RouteInput:
    return RouteInput(
        route_id=route.route_id,
        total_time_minutes=route.total_time_minutes,
        total_cost=route.total_cost,
        model_cost_unit=route.model_cost_unit,
        walking_distance_meters=route.walking_distance_meters,
        transfer_count=route.transfer_count,
    )


@router.post("/recommendations", response_model=RecommendationResponse)
def recommend(
    payload: RecommendationCreate,
    repository: Repository,
    settings: SettingsDependency,
) -> RecommendationResponse:
    command = RecommendationCommand(
        user_id=payload.user_id,
        ranking_mode=payload.ranking_mode,
        routes=tuple(_route_input(route) for route in payload.routes),
        constraints=ConstraintsInput(**payload.constraints.model_dump()),
        top_k=payload.top_k,
        jnd=(
            JndInput(
                shortlist_size=payload.jnd.shortlist_size,
                top_k=payload.jnd.top_k,
                **payload.jnd.thresholds.model_dump(),
            )
            if payload.jnd
            else None
        ),
    )
    result = RecommendationService(
        repository,
        model_cost_unit=settings.model_cost_unit,
    ).recommend(command=command)
    return RecommendationResponse(
        profile=present_profile(result.profile),
        ranking_mode=result.ranking_mode,
        candidate_count=result.candidate_count,
        feasible_count=result.feasible_count,
        ranked_routes=list(result.ranked_routes),
        rejected_routes=list(result.rejected_routes),
        explanation=result.explanation,
    )


@router.post("/users/{user_id}/choices", response_model=ChoiceLearningResponse)
def learn_choice(
    user_id: UUID,
    payload: ChoiceCreate,
    repository: Repository,
    settings: SettingsDependency,
) -> ChoiceLearningResponse:
    result = FeedbackService(
        repository,
        model_cost_unit=settings.model_cost_unit,
    ).learn_choice(
        command=ChoiceCommand(
            user_id=user_id,
            chosen_route=_route_input(payload.chosen_route),
            rejected_route=_route_input(payload.rejected_route),
        )
    )
    response = ChoiceLearningResponse(
        learning_applied=result.learning_applied,
        profile=present_profile(result.profile),
    )
    repository.commit()
    return response
