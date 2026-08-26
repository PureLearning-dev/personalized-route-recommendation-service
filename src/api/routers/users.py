from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Response, status

from ...application.records import ProfileRecord, UserRecord
from ...application.services import UserService
from ..dependencies import Repository
from ..presenters import present_profile
from ..schemas.profiles import ProfileResponse
from ..schemas.users import UserCreate, UserResponse

router = APIRouter(prefix="/users", tags=["users"])


def _user_response(user: UserRecord, profile: ProfileRecord) -> UserResponse:
    return UserResponse(
        id=user.id,
        external_user_id=user.external_user_id,
        initialization_mode=user.initialization_mode,
        preset_name=user.preset_name,
        created_at=user.created_at,
        profile=present_profile(profile),
    )


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(payload: UserCreate, repository: Repository) -> UserResponse:
    user, profile = UserService(repository).create_user(
        external_user_id=payload.external_user_id,
        initialization_mode=payload.initial_profile.mode,
        preset_name=payload.initial_profile.preset,
    )
    response = _user_response(user, profile)
    repository.commit()
    return response


@router.get("", response_model=list[UserResponse])
def list_users(repository: Repository) -> list[UserResponse]:
    return [_user_response(user, profile) for user, profile in UserService(repository).list_users()]


@router.get("/{user_id}/profile", response_model=ProfileResponse)
def current_profile(user_id: UUID, repository: Repository) -> ProfileResponse:
    return present_profile(UserService(repository).current_profile(user_id=user_id))


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: UUID, repository: Repository) -> Response:
    UserService(repository).delete_user(user_id=user_id)
    repository.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
