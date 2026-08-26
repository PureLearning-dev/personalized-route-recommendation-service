"""应用层所需的最小持久化端口。"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from ..profile.models import PreferenceLearningResult
from .records import ProfileRecord, UserRecord


class PersistencePort(Protocol):
    def commit(self) -> None: ...
    def rollback(self) -> None: ...

    def get_user(self, user_id: UUID) -> UserRecord | None: ...
    def get_user_by_external_id(self, external_user_id: str) -> UserRecord | None: ...
    def list_users_with_profiles(
        self,
    ) -> tuple[tuple[UserRecord, ProfileRecord], ...]: ...
    def add_user(
        self,
        external_user_id: str,
        initialization_mode: str,
        preset_name: str | None,
    ) -> UserRecord: ...
    def delete_user(self, user_id: UUID) -> bool: ...

    def add_profile(
        self,
        user_id: UUID,
        result: PreferenceLearningResult,
    ) -> ProfileRecord: ...
    def get_profile(self, user_id: UUID) -> ProfileRecord | None: ...
    def lock_profile(self, user_id: UUID) -> ProfileRecord | None: ...
    def update_profile(
        self,
        user_id: UUID,
        result: PreferenceLearningResult,
    ) -> ProfileRecord: ...
