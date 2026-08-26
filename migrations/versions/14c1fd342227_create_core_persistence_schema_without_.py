"""create minimal user and current profile schema

Revision ID: 14c1fd342227
Revises:
Create Date: 2026-08-25 11:12:21.120755
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "14c1fd342227"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("external_user_id", sa.String(length=200), nullable=False),
        sa.Column("initialization_mode", sa.String(length=16), nullable=False),
        sa.Column("preset_name", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "initialization_mode IN ('default', 'preset')",
            name="ck_user_init_mode",
        ),
        sa.CheckConstraint(
            "(initialization_mode = 'default' AND preset_name IS NULL) OR "
            "(initialization_mode = 'preset' AND preset_name IS NOT NULL)",
            name="ck_user_preset_shape",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("external_user_id"),
    )
    op.create_table(
        "user_profiles",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("coefficient_time", sa.Float(), nullable=False),
        sa.Column("coefficient_cost", sa.Float(), nullable=False),
        sa.Column("coefficient_walking", sa.Float(), nullable=False),
        sa.Column("coefficient_transfers", sa.Float(), nullable=False),
        sa.Column(
            "covariance",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("evidence_count", sa.Integer(), nullable=False),
        sa.Column("converged", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("evidence_count >= 0", name="ck_profile_evidence_count"),
        sa.CheckConstraint(
            "coefficient_time BETWEEN -20 AND 0",
            name="ck_profile_coefficient_time",
        ),
        sa.CheckConstraint(
            "coefficient_cost BETWEEN -20 AND 0",
            name="ck_profile_coefficient_cost",
        ),
        sa.CheckConstraint(
            "coefficient_walking BETWEEN -20 AND 0",
            name="ck_profile_coefficient_walking",
        ),
        sa.CheckConstraint(
            "coefficient_transfers BETWEEN -20 AND 0",
            name="ck_profile_coefficient_transfers",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id"),
    )


def downgrade() -> None:
    op.drop_table("user_profiles")
    op.drop_table("users")
