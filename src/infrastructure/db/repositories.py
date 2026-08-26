"""两张核心表的 SQLAlchemy/PostgreSQL 实现。"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from ...application.records import ProfileRecord, UserRecord, restore_learning_result
from ...profile.models import PreferenceDimension, PreferenceLearningResult
from .models import UserModel, UserProfileModel


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _user_record(model: UserModel) -> UserRecord:
    return UserRecord(
        id=model.id,
        external_user_id=model.external_user_id,
        initialization_mode=model.initialization_mode,
        preset_name=model.preset_name,
        created_at=model.created_at,
    )


def _profile_record(model: UserProfileModel) -> ProfileRecord:
    result = restore_learning_result(
        coefficients={
            PreferenceDimension.TIME.value: model.coefficient_time,
            PreferenceDimension.COST.value: model.coefficient_cost,
            PreferenceDimension.WALKING_DISTANCE.value: model.coefficient_walking,
            PreferenceDimension.TRANSFERS.value: model.coefficient_transfers,
        },
        covariance=tuple(tuple(row) for row in model.covariance),
        evidence_count=model.evidence_count,
        converged=model.converged,
    )
    return ProfileRecord(
        user_id=model.user_id,
        result=result,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def _apply_result(model: UserProfileModel, result: PreferenceLearningResult) -> None:
    coefficients = result.utility_coefficients
    model.coefficient_time = coefficients[PreferenceDimension.TIME]
    model.coefficient_cost = coefficients[PreferenceDimension.COST]
    model.coefficient_walking = coefficients[PreferenceDimension.WALKING_DISTANCE]
    model.coefficient_transfers = coefficients[PreferenceDimension.TRANSFERS]
    model.covariance = [list(row) for row in result.posterior.covariance]
    model.evidence_count = result.evidence_count
    model.converged = result.converged


class SqlAlchemyRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def commit(self) -> None:
        self.session.commit()

    def rollback(self) -> None:
        self.session.rollback()

    def get_user(self, user_id: UUID) -> UserRecord | None:
        model = self.session.get(UserModel, user_id)
        return _user_record(model) if model else None

    def get_user_by_external_id(self, external_user_id: str) -> UserRecord | None:
        model = self.session.scalar(
            select(UserModel).where(UserModel.external_user_id == external_user_id)
        )
        return _user_record(model) if model else None

    def list_users_with_profiles(
        self,
    ) -> tuple[tuple[UserRecord, ProfileRecord], ...]:
        rows = self.session.execute(
            select(UserModel, UserProfileModel)
            .join(UserProfileModel, UserProfileModel.user_id == UserModel.id)
            .order_by(UserModel.created_at, UserModel.id)
        ).all()
        return tuple((_user_record(user), _profile_record(profile)) for user, profile in rows)

    def add_user(
        self,
        external_user_id: str,
        initialization_mode: str,
        preset_name: str | None,
    ) -> UserRecord:
        model = UserModel(
            external_user_id=external_user_id,
            initialization_mode=initialization_mode,
            preset_name=preset_name,
        )
        self.session.add(model)
        self.session.flush()
        return _user_record(model)

    def delete_user(self, user_id: UUID) -> bool:
        model = self.session.get(UserModel, user_id)
        if model is None:
            return False
        self.session.delete(model)
        self.session.flush()
        return True

    def add_profile(
        self,
        user_id: UUID,
        result: PreferenceLearningResult,
    ) -> ProfileRecord:
        model = UserProfileModel(user_id=user_id)
        _apply_result(model, result)
        self.session.add(model)
        self.session.flush()
        return _profile_record(model)

    def get_profile(self, user_id: UUID) -> ProfileRecord | None:
        model = self.session.get(UserProfileModel, user_id)
        return _profile_record(model) if model else None

    def lock_profile(self, user_id: UUID) -> ProfileRecord | None:
        model = self.session.scalar(
            select(UserProfileModel).where(UserProfileModel.user_id == user_id).with_for_update()
        )
        return _profile_record(model) if model else None

    def update_profile(
        self,
        user_id: UUID,
        result: PreferenceLearningResult,
    ) -> ProfileRecord:
        model = self.session.get(UserProfileModel, user_id)
        if model is None:
            raise LookupError("用户画像不存在")
        _apply_result(model, result)
        model.updated_at = _utc_now()
        self.session.flush()
        return _profile_record(model)
