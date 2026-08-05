"""通过设计好的路线比较题，交互式反推用户的长期偏好权重。

本程序只验证一件事：用户依次比较若干组具有明显取舍的路线，正式画像学习器
根据选择结果反推出用户对时间、费用、步行距离和换乘次数的敏感程度。

比较题中的路线属性是问卷设计的一部分，并不是仅用于展示的模拟结果。用户
选择 A 或 B 后，两条路线的四项属性都会进入 ``PairwisePreferenceWeightLearner``
参与计算。程序不收集硬约束、本次出行情况或实时环境，也不执行路线排序。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
import sys

# 支持在项目根目录直接运行本文件。这里只处理示例程序的导入路径，不包含
# 任何画像计算；真正的权重反推始终由 src.profile 中的业务实现完成。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.profile import (  # noqa: E402
    PREFERENCE_DIMENSIONS,
    PairwisePreference,
    PairwisePreferenceWeightLearner,
    PreferenceDimension,
    RouteAttributes,
)


# 把终端输入输出作为参数传入，既可以供用户真实体验，也便于自动化测试按照
# 同样的交互顺序提供选择，而不需要复制一份画像学习逻辑。
InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]


DIMENSION_LABELS: Mapping[PreferenceDimension, str] = {
    PreferenceDimension.TIME: "时间",
    PreferenceDimension.COST: "费用",
    PreferenceDimension.WALKING_DISTANCE: "步行距离",
    PreferenceDimension.TRANSFERS: "换乘次数",
}


@dataclass(frozen=True, slots=True)
class ComparisonScenario:
    """一组已经设计好的路线比较题。

    每题主要制造两个指标之间的取舍，并尽量保持另外两个指标相同。例如
    “时间与费用”题只让时间和费用明显变化，从而使用户选择更容易解释。
    """

    title: str
    route_a: RouteAttributes
    route_b: RouteAttributes


def _comparison_scenarios() -> tuple[ComparisonScenario, ...]:
    """返回覆盖四项指标两两组合的6组路线比较题。

    四个指标一共有6种两两组合，因此这里用6题让每个指标都能与其他指标
    发生取舍。这些数值会真正进入学习器，不是只在终端中展示。
    """

    return (
        ComparisonScenario(
            "时间与费用",
            RouteAttributes("q1-a", 30, 40, 400, 1),
            RouteAttributes("q1-b", 60, 8, 400, 1),
        ),
        ComparisonScenario(
            "时间与步行距离",
            RouteAttributes("q2-a", 30, 20, 1200, 1),
            RouteAttributes("q2-b", 55, 20, 200, 1),
        ),
        ComparisonScenario(
            "时间与换乘次数",
            RouteAttributes("q3-a", 30, 20, 400, 2),
            RouteAttributes("q3-b", 55, 20, 400, 0),
        ),
        ComparisonScenario(
            "费用与步行距离",
            RouteAttributes("q4-a", 45, 8, 1200, 1),
            RouteAttributes("q4-b", 45, 35, 200, 1),
        ),
        ComparisonScenario(
            "费用与换乘次数",
            RouteAttributes("q5-a", 45, 8, 400, 2),
            RouteAttributes("q5-b", 45, 35, 400, 0),
        ),
        ComparisonScenario(
            "步行距离与换乘次数",
            RouteAttributes("q6-a", 45, 20, 200, 2),
            RouteAttributes("q6-b", 45, 20, 1200, 0),
        ),
    )


def _format_route(route: RouteAttributes) -> str:
    """用统一格式展示学习器实际接收的四项路线属性。"""

    return (
        f"时间 {route.total_time_minutes:g} 分钟｜费用 {route.total_cost:g} 元｜"
        f"步行 {route.walking_distance_meters:g} 米｜换乘 {route.transfer_count} 次"
    )


def _ask_preference(
    input_fn: InputFunction,
    output_fn: OutputFunction,
    scenario: ComparisonScenario,
) -> PairwisePreference | None:
    """读取一题选择，并转换为学习器需要的“已选路线优于未选路线”。"""

    output_fn(f"主要比较：{scenario.title}")
    output_fn(f"A：{_format_route(scenario.route_a)}")
    output_fn(f"B：{_format_route(scenario.route_b)}")

    while True:
        answer = input_fn("你更愿意选择 A 还是 B？输入 S 可跳过：").strip().lower()
        if answer in {"a", "b", "s", "skip", "跳过"}:
            break
        output_fn("请输入 A、B 或 S。")

    if answer in {"s", "skip", "跳过"}:
        output_fn("本题已跳过，不会形成偏好证据。")
        return None

    chosen = scenario.route_a if answer == "a" else scenario.route_b
    rejected = scenario.route_b if answer == "a" else scenario.route_a
    return PairwisePreference(chosen=chosen, rejected=rejected)


def _print_result(
    output_fn: OutputFunction,
    comparisons: tuple[PairwisePreference, ...],
) -> None:
    """调用正式学习器反推权重，并展示结果及基本质量信息。"""

    # 没有历史群体数据时，按论文图2从 N(0, I) 的MPP初值开始。
    learner = PairwisePreferenceWeightLearner()
    result = learner.fit(comparisons)

    output_fn("\n根据路线选择反推得到的四维偏好画像")
    for dimension in PREFERENCE_DIMENSIONS:
        standard_deviation = result.standard_deviations[dimension]
        output_fn(
            f"  {DIMENSION_LABELS[dimension]}：{result.weights[dimension]:.2%}"
            f"（论文效用系数 {result.utility_coefficients[dimension]:.4f}，"
            f"后验标准差 {standard_deviation:.4f}）"
        )

    most_important = max(PREFERENCE_DIMENSIONS, key=result.weights.__getitem__)
    output_fn(
        f"  当前最敏感的指标：{DIMENSION_LABELS[most_important]}"
        "（越敏感，越不愿意承受该项代价）"
    )
    output_fn(f"  累计有效路线比较：{result.evidence_count} 条")
    if result.choice_probabilities:
        mean_probability = sum(result.choice_probabilities) / len(
            result.choice_probabilities
        )
        output_fn(f"  已选路线的平均后验概率：{mean_probability:.2%}")
    output_fn("  注：后验标准差越大，表示当前证据下该维度仍越不确定。")


def run_interactive(
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
) -> None:
    """执行“设计路线比较—收集选择—反推权重”的完整交互流程。"""

    # 生成 6 组路线
    scenarios = _comparison_scenarios()
    output_fn("=" * 64)
    output_fn("个性化多模式出行——路线选择反推偏好权重")
    output_fn(f"下面共有 {len(scenarios)} 组设计好的路线，请根据真实偏好选择 A 或 B。")
    output_fn("你的选择会与路线的时间、费用、步行和换乘属性一起进入学习器。")
    output_fn("=" * 64)

    # collected 用于保存测试时选择和拒绝的路线属性信息
    collected: list[PairwisePreference] = []
    for index, scenario in enumerate(scenarios, start=1):
        output_fn(f"\n[{index}/{len(scenarios)}]")
        preference = _ask_preference(input_fn, output_fn, scenario)
        if preference is not None:
            collected.append(preference)

    comparisons = tuple(collected)
    if not comparisons:
        output_fn("\n没有有效选择，系统返回四项等权的 Gaussian 初始画像。")
    else:
        output_fn(f"\n已收集 {len(comparisons)} 条有效选择，开始反推个人偏好权重。")

    # 通过得到的选择计算后得到偏好权重
    _print_result(output_fn, comparisons)


def main() -> None:
    """命令行入口，并在用户主动结束输入时安全退出。"""

    try:
        run_interactive()
    except (EOFError, KeyboardInterrupt):
        print("\n输入已结束，交互程序安全退出。")


if __name__ == "__main__":
    main()
