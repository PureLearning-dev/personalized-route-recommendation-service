"""把FAVOUR学习到的四维画像用于候选路线Top-K排序。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.profile import (  # noqa: E402
    PREFERENCE_DIMENSIONS,
    PairwisePreference,
    PairwisePreferenceWeightLearner,
    PreferenceDimension,
    PreferencePreset,
    RouteAttributes,
)
from src.recommendation import PersonalizedRouteRanker, RouteConstraints  # noqa: E402


DIMENSION_LABELS: Mapping[PreferenceDimension, str] = {
    PreferenceDimension.TIME: "时间",
    PreferenceDimension.COST: "费用",
    PreferenceDimension.WALKING_DISTANCE: "步行距离",
    PreferenceDimension.TRANSFERS: "换乘次数",
}


def _cost_sensitive_comparisons() -> tuple[PairwisePreference, ...]:
    return tuple(
        PairwisePreference(
            chosen=RouteAttributes(f"history-cheap-{index}", 70, 8, 400, 1),
            rejected=RouteAttributes(f"history-fast-{index}", 35, 55, 400, 1),
        )
        for index in range(1, 3)
    )


def main() -> None:
    # 先使用业务侧的费用优先画像，再让两次真实路线选择修正该先验。
    profile = PairwisePreferenceWeightLearner().fit(
        _cost_sensitive_comparisons(),
        preference_preset=PreferencePreset.COST_PRIORITY,
    )
    candidates = (
        RouteAttributes("fast-expensive", 32, 68, 350, 1),
        RouteAttributes("balanced", 50, 36, 450, 1),
        RouteAttributes("cheap-slow", 78, 12, 600, 1),
        RouteAttributes("too-much-walking", 45, 18, 2200, 0),
    )
    result = PersonalizedRouteRanker().rank(
        candidates,
        profile,
        constraints=RouteConstraints(max_walking_distance_meters=1500),
        top_k=3,
    )

    print("FAVOUR画像权重")
    for dimension in PREFERENCE_DIMENSIONS:
        print(f"  {DIMENSION_LABELS[dimension]}：{profile.weights[dimension]:.1%}")

    print("\n个性化路线排序（加权代价越低越好）")
    for item in result.ranked_routes:
        advantages = "、".join(
            DIMENSION_LABELS[dimension]
            for dimension in item.advantage_dimensions[:2]
        ) or "综合表现"
        print(
            f"  {item.rank}. {item.route.route_id}｜"
            f"加权代价 {item.personalized_cost:.4f}｜主要相对优势：{advantages}"
        )

    print("\n硬约束排除")
    for item in result.rejected_routes:
        labels = "、".join(DIMENSION_LABELS[d] for d in item.violated_dimensions)
        print(f"  {item.route.route_id}：超过{labels}上限")


if __name__ == "__main__":
    main()
