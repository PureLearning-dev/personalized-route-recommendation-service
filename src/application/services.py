"""用户画像、即时推荐和同步学习服务。"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from ..profile.learner import PairwisePreferenceWeightLearner
from ..profile.models import (
    PREFERENCE_DIMENSIONS,
    PairwisePreference,
    PreferenceDimension,
    PreferencePreset,
    RouteAttributes,
)
from ..recommendation import (
    JndEnhancedRouteRanker,
    JndThresholds,
    PersonalizedRouteRanker,
    RouteConstraints,
)
from ..recommendation.models import JndEnhancedRankingResult, RouteRankingResult
from .commands import ChoiceCommand, RecommendationCommand, RouteInput
from .errors import ConflictError, NotFoundError, ValidationError
from .ports import PersistencePort
from .records import (
    ChoiceLearningRecord,
    ProfileRecord,
    RecommendationResultRecord,
    UserRecord,
)


def _dimension_map(values: Any) -> dict[str, float]:
    return {dimension.value: float(values[dimension]) for dimension in PreferenceDimension}


def _route_attributes(item: RouteInput) -> RouteAttributes:
    return RouteAttributes(
        route_id=item.route_id,
        total_time_minutes=item.total_time_minutes,
        total_cost=item.total_cost,
        walking_distance_meters=item.walking_distance_meters,
        transfer_count=item.transfer_count,
    )


def _ranked_route(item: Any) -> dict[str, Any]:
    return {
        "rank": item.rank,
        "route": {
            "route_id": item.route.route_id,
            "total_time_minutes": item.route.total_time_minutes,
            "total_cost": item.route.total_cost,
            "walking_distance_meters": item.route.walking_distance_meters,
            "transfer_count": item.route.transfer_count,
        },
        "personalized_cost": item.personalized_cost,
        "normalized_attributes": _dimension_map(item.normalized_attributes),
        "weighted_contributions": _dimension_map(item.weighted_contributions),
        "advantage_dimensions": [dimension.value for dimension in item.advantage_dimensions],
    }


def _rejected_route(item: Any) -> dict[str, Any]:
    return {
        "route_id": item.route.route_id,
        "violated_dimensions": [dimension.value for dimension in item.violated_dimensions],
    }


class UserService:
    def __init__(
        self,
        repository: PersistencePort,
        *,
        learner: PairwisePreferenceWeightLearner | None = None,
    ) -> None:
        self.repository = repository
        self.learner = learner or PairwisePreferenceWeightLearner()

    def create_user(
        self,
        *,
        external_user_id: str,
        initialization_mode: str,
        preset_name: str | None,
    ) -> tuple[UserRecord, ProfileRecord]:
        external_user_id = external_user_id.strip()
        if not external_user_id:
            raise ValidationError("external_user_id 不能为空")
        if self.repository.get_user_by_external_id(external_user_id):
            raise ConflictError("该 external_user_id 已存在")

        if initialization_mode == "default":
            if preset_name is not None:
                raise ValidationError("default 初始化不能携带 preset")
            preset = None
        elif initialization_mode == "preset":
            if preset_name is None:
                raise ValidationError("preset 初始化必须指定 preset")
            try:
                preset = PreferencePreset(preset_name)
            except ValueError as error:
                raise ValidationError("未知命名预设") from error
        else:
            raise ValidationError("initialization_mode 必须是 default 或 preset")

        result = self.learner.fit((), preference_preset=preset)
        user = self.repository.add_user(
            external_user_id,
            initialization_mode,
            preset.value if preset else None,
        )
        return user, self.repository.add_profile(user.id, result)

    def current_profile(self, *, user_id: UUID) -> ProfileRecord:
        if self.repository.get_user(user_id) is None:
            raise NotFoundError("用户不存在")
        profile = self.repository.get_profile(user_id)
        if profile is None:
            raise NotFoundError("用户画像不存在")
        return profile

    def list_users(self) -> tuple[tuple[UserRecord, ProfileRecord], ...]:
        return self.repository.list_users_with_profiles()

    def delete_user(self, *, user_id: UUID) -> None:
        if not self.repository.delete_user(user_id):
            raise NotFoundError("用户不存在")


class RecommendationService:
    def __init__(
        self,
        repository: PersistencePort,
        *,
        model_cost_unit: str,
    ) -> None:
        self.repository = repository
        self.model_cost_unit = model_cost_unit
        self.weighted_ranker = PersonalizedRouteRanker()
        self.jnd_ranker = JndEnhancedRouteRanker(self.weighted_ranker)

    def recommend(self, *, command: RecommendationCommand) -> RecommendationResultRecord:
        profile = UserService(self.repository).current_profile(user_id=command.user_id)
        for route in command.routes:
            if route.model_cost_unit != self.model_cost_unit:
                raise ValidationError(f"费用单位必须统一为 {self.model_cost_unit}")

        routes = tuple(_route_attributes(item) for item in command.routes)
        constraints = RouteConstraints(
            max_total_time_minutes=command.constraints.max_total_time_minutes,
            max_total_cost=command.constraints.max_total_cost,
            max_walking_distance_meters=command.constraints.max_walking_distance_meters,
            max_transfer_count=command.constraints.max_transfer_count,
        )

        if command.ranking_mode == "weighted":
            if command.jnd is not None:
                raise ValidationError("weighted 模式不能携带 jnd 参数")
            result: RouteRankingResult | JndEnhancedRankingResult = self.weighted_ranker.rank(
                routes,
                profile.result,
                constraints=constraints,
                top_k=command.top_k,
            )
            explanation = {"normalized_weights": _dimension_map(result.normalized_weights)}
        elif command.ranking_mode == "jnd":
            if command.jnd is None:
                raise ValidationError("jnd 模式必须提供完整 jnd 参数")
            thresholds = JndThresholds(
                time_ratio=command.jnd.time_ratio,
                cost_ratio=command.jnd.cost_ratio,
                walking_distance_ratio=command.jnd.walking_distance_ratio,
                transfers_ratio=command.jnd.transfers_ratio,
            )
            result = self.jnd_ranker.rank(
                routes,
                profile.result,
                thresholds=thresholds,
                shortlist_size=command.jnd.shortlist_size,
                top_k=command.jnd.top_k,
                constraints=constraints,
            )
            explanation = {
                "normalized_weights": _dimension_map(result.normalized_weights),
                "attribute_priority": [dimension.value for dimension in result.attribute_priority],
                "reference_values": (
                    _dimension_map(result.reference_values) if result.reference_values else {}
                ),
                "comparison_steps": [
                    {
                        "priority_level": step.priority_level,
                        "dimension": step.dimension.value,
                        "route_ids": list(step.route_ids),
                        "reference_value": step.reference_value,
                        "threshold_ratio": step.threshold_ratio,
                        "noticeable_difference": step.noticeable_difference,
                        "within_jnd_route_ids": list(step.within_jnd_route_ids),
                        "outside_jnd_route_ids": list(step.outside_jnd_route_ids),
                    }
                    for step in result.comparison_steps
                ],
            }
        else:
            raise ValidationError("ranking_mode 必须是 weighted 或 jnd")

        return RecommendationResultRecord(
            profile=profile,
            ranking_mode=command.ranking_mode,
            candidate_count=result.candidate_count,
            feasible_count=result.feasible_count,
            ranked_routes=tuple(_ranked_route(item) for item in result.ranked_routes),
            rejected_routes=tuple(_rejected_route(item) for item in result.rejected_routes),
            explanation=explanation,
        )


class FeedbackService:
    def __init__(
        self,
        repository: PersistencePort,
        *,
        model_cost_unit: str,
        learner: PairwisePreferenceWeightLearner | None = None,
    ) -> None:
        self.repository = repository
        self.model_cost_unit = model_cost_unit
        self.learner = learner or PairwisePreferenceWeightLearner()

    def learn_choice(self, *, command: ChoiceCommand) -> ChoiceLearningRecord:
        if self.repository.get_user(command.user_id) is None:
            raise NotFoundError("用户不存在")
        for route in (command.chosen_route, command.rejected_route):
            if route.model_cost_unit != self.model_cost_unit:
                raise ValidationError(f"费用单位必须统一为 {self.model_cost_unit}")
        if command.chosen_route.route_id == command.rejected_route.route_id:
            raise ValidationError("选中路线和对比路线必须是两条不同路线")

        profile = self.repository.lock_profile(command.user_id)
        if profile is None:
            raise NotFoundError("用户画像不存在")
        chosen = _route_attributes(command.chosen_route)
        rejected = _route_attributes(command.rejected_route)
        if all(
            chosen.value_for(dimension) == rejected.value_for(dimension)
            for dimension in PREFERENCE_DIMENSIONS
        ):
            return ChoiceLearningRecord(profile=profile, learning_applied=False)

        result = self.learner.update(
            profile.result,
            (PairwisePreference(chosen=chosen, rejected=rejected),),
        )
        updated = self.repository.update_profile(command.user_id, result)
        return ChoiceLearningRecord(profile=updated, learning_applied=True)
