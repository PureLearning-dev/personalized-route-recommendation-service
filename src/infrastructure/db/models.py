"""核心用户与当前画像的 SQLAlchemy 映射。"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


def utc_now() -> datetime:
    return datetime.now(UTC)


class UserModel(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    external_user_id: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    initialization_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    preset_name: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    __table_args__ = (
        CheckConstraint("initialization_mode IN ('default', 'preset')", name="ck_user_init_mode"),
        CheckConstraint(
            "(initialization_mode = 'default' AND preset_name IS NULL) OR "
            "(initialization_mode = 'preset' AND preset_name IS NOT NULL)",
            name="ck_user_preset_shape",
        ),
    )


class UserProfileModel(Base):
    __tablename__ = "user_profiles"

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    coefficient_time: Mapped[float] = mapped_column(Float, nullable=False)
    coefficient_cost: Mapped[float] = mapped_column(Float, nullable=False)
    coefficient_walking: Mapped[float] = mapped_column(Float, nullable=False)
    coefficient_transfers: Mapped[float] = mapped_column(Float, nullable=False)
    covariance: Mapped[list[list[float]]] = mapped_column(JSONB, nullable=False)
    evidence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    converged: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    __table_args__ = (
        CheckConstraint("evidence_count >= 0", name="ck_profile_evidence_count"),
        CheckConstraint("coefficient_time BETWEEN -20 AND 0", name="ck_profile_coefficient_time"),
        CheckConstraint("coefficient_cost BETWEEN -20 AND 0", name="ck_profile_coefficient_cost"),
        CheckConstraint(
            "coefficient_walking BETWEEN -20 AND 0",
            name="ck_profile_coefficient_walking",
        ),
        CheckConstraint(
            "coefficient_transfers BETWEEN -20 AND 0",
            name="ck_profile_coefficient_transfers",
        ),
    )
