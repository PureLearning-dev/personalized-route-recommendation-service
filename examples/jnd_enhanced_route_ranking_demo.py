"""详细展示“画像学习—加权初排—JND精排—Top-K”的验证过程。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    PreferenceLearningResult,
    RouteAttributes,
)
from src.recommendation import (  # noqa: E402
    JndEnhancedRankingResult,
    JndEnhancedRouteRanker,
    JndThresholds,
    RankedRoute,
    RouteConstraints,
)


DIMENSION_LABELS: Mapping[PreferenceDimension, str] = {
    PreferenceDimension.TIME: "时间",
    PreferenceDimension.COST: "费用",
    PreferenceDimension.WALKING_DISTANCE: "步行距离",
    PreferenceDimension.TRANSFERS: "换乘次数",
}


@dataclass(frozen=True, slots=True)
class FixedRouteChoice:
    """演示中写在代码里的一个固定选择，A和B仍会完整打印。"""

    title: str
    route_a: RouteAttributes
    route_b: RouteAttributes
    selected: str

    @property
    def comparison(self) -> PairwisePreference:
        chosen = self.route_a if self.selected == "A" else self.route_b
        rejected = self.route_b if self.selected == "A" else self.route_a
        return PairwisePreference(chosen=chosen, rejected=rejected)


def _fixed_profile_choices() -> tuple[FixedRouteChoice, ...]:
    """构造覆盖四个指标两两取舍的六次固定路线选择。"""

    return (
        FixedRouteChoice(
            "时间与费用",
            RouteAttributes("q1-a", 30, 40, 400, 1),
            RouteAttributes("q1-b", 60, 8, 400, 1),
            "A",
        ),
        FixedRouteChoice(
            "时间与步行距离",
            RouteAttributes("q2-a", 30, 20, 1200, 1),
            RouteAttributes("q2-b", 55, 20, 200, 1),
            "A",
        ),
        FixedRouteChoice(
            "时间与换乘次数",
            RouteAttributes("q3-a", 30, 20, 400, 2),
            RouteAttributes("q3-b", 55, 20, 400, 0),
            "A",
        ),
        FixedRouteChoice(
            "费用与步行距离",
            RouteAttributes("q4-a", 45, 8, 1200, 1),
            RouteAttributes("q4-b", 45, 35, 200, 1),
            "A",
        ),
        FixedRouteChoice(
            "费用与换乘次数",
            RouteAttributes("q5-a", 45, 8, 400, 2),
            RouteAttributes("q5-b", 45, 35, 400, 0),
            "A",
        ),
        FixedRouteChoice(
            "步行距离与换乘次数",
            RouteAttributes("q6-a", 45, 20, 200, 2),
            RouteAttributes("q6-b", 45, 20, 1200, 0),
            "A",
        ),
    )


def _candidate_routes() -> tuple[RouteAttributes, ...]:
    return (
        RouteAttributes("fast-expensive", 30, 50, 100, 1),
        RouteAttributes("slow-cheap", 40, 40, 100, 1),
        RouteAttributes("balanced", 33, 45, 100, 1),
        RouteAttributes("too-much-walking", 28, 42, 2200, 0),
    )


def _format_route(route: RouteAttributes) -> str:
    return (
        f"时间 {route.total_time_minutes:g} 分钟｜费用 {route.total_cost:g} 元｜"
        f"步行 {route.walking_distance_meters:g} 米｜换乘 {route.transfer_count} 次"
    )


def _format_weights(profile: PreferenceLearningResult) -> str:
    return "｜".join(
        f"{DIMENSION_LABELS[dimension]} {profile.weights[dimension]:.1%}"
        for dimension in PREFERENCE_DIMENSIONS
    )


def _format_route_ids(routes: Sequence[RankedRoute]) -> str:
    return " → ".join(item.route.route_id for item in routes)


def _learn_and_print_profile() -> PreferenceLearningResult:
    print("=" * 72)
    print("步骤1：展示代码中预设的路线选择，并逐步学习用户画像")
    print("=" * 72)
    learner = PairwisePreferenceWeightLearner()
    comparisons: list[PairwisePreference] = []

    for index, choice in enumerate(_fixed_profile_choices(), start=1):
        comparison = choice.comparison
        comparisons.append(comparison)
        current_profile = learner.fit(tuple(comparisons))

        print(f"\n第{index}题：{choice.title}")
        print(f"  路线A：{_format_route(choice.route_a)}")
        print(f"  路线B：{_format_route(choice.route_b)}")
        print(
            f"  代码预设选择：{choice.selected}，即选择 "
            f"{comparison.chosen.route_id}，放弃 {comparison.rejected.route_id}"
        )
        print(f"  累计{index}次选择后的画像：{_format_weights(current_profile)}")

    profile = learner.fit(tuple(comparisons))
    print("\n最终画像")
    print(f"  有效选择数量：{profile.evidence_count}")
    print(f"  学习是否收敛：{'是' if profile.converged else '否'}")
    print(f"  最终权重：{_format_weights(profile)}")
    return profile


def _print_candidates_and_constraints(
    candidates: Sequence[RouteAttributes],
    constraints: RouteConstraints,
    result: JndEnhancedRankingResult,
) -> None:
    print("\n" + "=" * 72)
    print("步骤2：展示本次推荐的候选路线和硬约束过滤")
    print("=" * 72)
    print(
        "本次约束："
        f"总时间≤{constraints.max_total_time_minutes:g}分钟｜"
        f"费用≤{constraints.max_total_cost:g}元｜"
        f"步行≤{constraints.max_walking_distance_meters:g}米｜"
        f"换乘≤{constraints.max_transfer_count}次"
    )
    rejected = {item.route.route_id: item for item in result.rejected_routes}
    for route in candidates:
        rejected_route = rejected.get(route.route_id)
        if rejected_route is None:
            status = "通过硬约束"
        else:
            reasons = "、".join(
                DIMENSION_LABELS[dimension]
                for dimension in rejected_route.violated_dimensions
            )
            status = f"被过滤：超过{reasons}上限"
        print(f"  {route.route_id}｜{_format_route(route)}｜{status}")


def _print_weighted_ranking(result: JndEnhancedRankingResult) -> None:
    print("\n" + "=" * 72)
    print("步骤3：展示每条可行路线的归一化、分项贡献和加权初排")
    print("=" * 72)
    weights = result.normalized_weights

    for item in result.weighted_result.ranked_routes:
        print(f"\n路线：{item.route.route_id}")
        for dimension in PREFERENCE_DIMENSIONS:
            print(
                f"  {DIMENSION_LABELS[dimension]}："
                f"原始值 {item.route.value_for(dimension):g} → "
                f"归一化 {item.normalized_attributes[dimension]:.4f} × "
                f"权重 {weights[dimension]:.1%} = "
                f"分项代价 {item.weighted_contributions[dimension]:.4f}"
            )
        print(f"  个性化总代价：{item.personalized_cost:.4f}")

    print("\n加权代价越小，初排越靠前：")
    print(f"  {_format_route_ids(result.weighted_result.ranked_routes)}")
    shortlist = result.weighted_result.ranked_routes[: result.shortlist_size]
    print(f"进入JND精排的Top-{result.shortlist_size}：{_format_route_ids(shortlist)}")


def _print_jnd_process(result: JndEnhancedRankingResult) -> None:
    print("\n" + "=" * 72)
    print("步骤4：展示JND逐层比较的中间过程")
    print("=" * 72)
    priority = " → ".join(
        f"{DIMENSION_LABELS[dimension]}({result.normalized_weights[dimension]:.1%})"
        for dimension in result.attribute_priority
    )
    print(f"指标优先级由画像权重从高到低确定：{priority}")

    print("\nTop-N共同参考值和JND范围：")
    for dimension in result.attribute_priority:
        reference = result.reference_values[dimension]
        ratio = result.thresholds.ratio_for(dimension)
        difference = reference * ratio
        print(
            f"  {DIMENSION_LABELS[dimension]}：最优参考值 {reference:g}，"
            f"阈值 {ratio:.1%}，允许差值 {difference:g}，"
            f"JND最优范围≤{reference + difference:g}"
        )

    for index, step in enumerate(result.comparison_steps, start=1):
        label = DIMENSION_LABELS[step.dimension]
        print(f"\n第{index}轮：比较 {'、'.join(step.route_ids)}")
        print(
            f"  当前指标：{label}（优先级第{step.priority_level}）｜"
            f"共同最优值 {step.reference_value:g}｜"
            f"可感知差值必须大于 {step.noticeable_difference:g}"
        )
        inside = "、".join(step.within_jnd_route_ids) or "无"
        outside = "、".join(step.outside_jnd_route_ids) or "无"
        print(f"  差异不明显，进入下一指标继续比较：{inside}")
        print(f"  差异已经明显，可由当前指标区分：{outside}")
        if step.outside_jnd_route_ids:
            print("  本轮规则：JND范围内路线优先，范围外路线按当前代价升序排列。")
        else:
            print("  本轮无法区分路线，整个比较组进入下一指标。")


def _print_final_result(result: JndEnhancedRankingResult) -> None:
    print("\n" + "=" * 72)
    print("步骤5：输出最终Top-K路线，供用户选择")
    print("=" * 72)
    print(
        f"JND精排顺序：{_format_route_ids(result.ranked_routes)}\n"
        f"最终推荐Top-{len(result.ranked_routes)}如下："
    )
    for item in result.ranked_routes:
        dimension = result.decisive_dimensions[item.route.route_id]
        reason = (
            f"首次落在JND范围外的指标是{DIMENSION_LABELS[dimension]}"
            if dimension is not None
            else "比较过程中始终留在JND最优组"
        )
        print(f"\n  推荐路线{item.rank}：{item.route.route_id}")
        print(f"    路线信息：{_format_route(item.route)}")
        print(f"    加权代价：{item.personalized_cost:.4f}")
        print(f"    JND说明：{reason}")

    # 演示代码没有真实交互界面，因此固定模拟用户选择第一条推荐路线。
    # 实际系统应在这里接收用户对Top-K的真实选择并记录为后续反馈。
    selected = result.ranked_routes[0]
    print("\n模拟用户选择")
    print(
        f"  用户从Top-{len(result.ranked_routes)}中选择："
        f"推荐路线{selected.rank}（{selected.route.route_id}）"
    )
    print("  该选择可在后续作为更新画像权重和个性化JND阈值的反馈数据。")


def main() -> None:
    profile = _learn_and_print_profile()
    candidates = _candidate_routes()
    constraints = RouteConstraints(
        max_total_time_minutes=60,
        max_total_cost=60,
        max_walking_distance_meters=1500,
        max_transfer_count=2,
    )
    thresholds = JndThresholds(
        time_ratio=0.20,
        cost_ratio=0.20,
        walking_distance_ratio=0.15,
        transfers_ratio=0.0,
    )
    result = JndEnhancedRouteRanker().rank(
        candidates,
        profile,
        constraints=constraints,
        thresholds=thresholds,
        shortlist_size=3,
        top_k=2,
    )

    _print_candidates_and_constraints(candidates, constraints, result)
    _print_weighted_ranking(result)
    _print_jnd_process(result)
    _print_final_result(result)


if __name__ == "__main__":
    main()
