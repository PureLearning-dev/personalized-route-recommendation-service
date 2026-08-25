"""输出 JND 核心排序过程，便于理解和截图。"""

from __future__ import annotations

from collections.abc import Sequence

from src.profile import PreferenceDimension, RouteAttributes
from src.recommendation import JndEnhancedRouteRanker, JndThresholds, RankedRoute


DIMENSION_LABELS = {
    PreferenceDimension.TIME: "时间",
    PreferenceDimension.COST: "费用",
    PreferenceDimension.WALKING_DISTANCE: "步行",
    PreferenceDimension.TRANSFERS: "换乘",
}


def _route_ids(items: Sequence[RankedRoute]) -> str:
    return " → ".join(item.route.route_id for item in items)


def main() -> None:
    routes = (
        RouteAttributes("fast-expensive", 30, 80, 100, 1),
        RouteAttributes("slow-cheap", 40, 10, 100, 1),
        RouteAttributes("balanced", 33, 50, 100, 1),
    )
    weights = {
        PreferenceDimension.TIME: 0.6,
        PreferenceDimension.COST: 0.4,
        PreferenceDimension.WALKING_DISTANCE: 0.0,
        PreferenceDimension.TRANSFERS: 0.0,
    }
    thresholds = JndThresholds(
        time_ratio=0.2,
        cost_ratio=0.1,
        walking_distance_ratio=0.0,
        transfers_ratio=0.0,
    )

    result = JndEnhancedRouteRanker().rank(
        routes,
        weights,
        thresholds=thresholds,
        shortlist_size=3,
        top_k=3,
    )
    routes_by_id = {route.route_id: route for route in routes}

    print("=" * 72)
    print("JND 核心逻辑测试：加权初排 → 共同参考值 → JND 精排")
    print("=" * 72)
    print("参数：时间权重 0.6｜费用权重 0.4｜时间 JND 20%｜费用 JND 10%")

    print("\n[1] 加权初排（综合代价越低越好）")
    for item in result.weighted_result.ranked_routes:
        time_part = item.weighted_contributions[PreferenceDimension.TIME]
        cost_part = item.weighted_contributions[PreferenceDimension.COST]
        print(
            f"  {item.rank}. {item.route.route_id:<16} "
            f"时间贡献={time_part:.4f} 费用贡献={cost_part:.4f} "
            f"综合代价={item.personalized_cost:.4f}"
        )
    print(f"  初排结果：{_route_ids(result.weighted_result.ranked_routes)}")

    priority = " → ".join(
        DIMENSION_LABELS[dimension] for dimension in result.attribute_priority
    )
    print(f"\n[2] JND 指标优先级：{priority}")
    print("    权重决定比较顺序；JND 使用路线原始属性值进行比较。")

    print("\n[3] JND 分组过程")
    for step in result.comparison_steps:
        label = DIMENSION_LABELS[step.dimension]
        print(
            f"  第 {step.priority_level} 轮｜{label}｜共同参考值={step.reference_value:g} "
            f"｜阈值={step.threshold_ratio:.0%} "
            f"｜可感知界限={step.noticeable_difference:g}"
        )
        for route_id in step.route_ids:
            value = routes_by_id[route_id].value_for(step.dimension)
            difference = value - step.reference_value
            relation = (
                "范围内，继续比较下一指标"
                if route_id in step.within_jnd_route_ids
                else "范围外，当前差异已经明显"
            )
            print(
                f"      {route_id:<16} 原始值={value:g} "
                f"差值={difference:g} → {relation}"
            )
        within = "、".join(step.within_jnd_route_ids) or "无"
        outside = "、".join(step.outside_jnd_route_ids) or "无"
        print(f"      JND 范围内：{within}")
        print(f"      JND 范围外：{outside}")

    print("\n[4] JND 精排结果")
    print(f"  {_route_ids(result.ranked_routes)}")
    print("  balanced 与 fast-expensive 的时间差不明显，因此继续比较费用；")
    print("  balanced 费用更低，所以排在 fast-expensive 前面；")
    print("  slow-cheap 慢 10 分钟，超过 6 分钟时间阈值，因此被降到最后。")

    print("\n测试结论：JND 在保留加权初排候选集的基础上，优先处理用户真正")
    print("          关注且能够感知的差异，使最终推荐更符合用户实际体验。")
    print("=" * 72)


if __name__ == "__main__":
    main()
