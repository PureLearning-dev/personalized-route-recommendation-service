"""用预设路线选择自动验证四维个人画像学习流程。

该示例不读取终端输入。程序会先用固定随机种子生成较多路线对，再按照四类
明确的目标偏好自动形成 ``PairwisePreference`` 列表，并把列表一次性传给
正式的 ``PairwisePreferenceWeightLearner``。如果最终画像不能识别预设的
主偏好，或者拟合质量过低，程序会抛出 ``AssertionError`` 并以非零状态退出。

运行方式：

    PYTHONDONTWRITEBYTECODE=1 python3 examples/automated_profile_validation.py

默认展示每类画像前五条完整路线选择；查看全部选择可运行：

    PYTHONDONTWRITEBYTECODE=1 python3 examples/automated_profile_validation.py --show-all
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random
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
from src.profile.normalization import NORMALIZATION_SCALES  # noqa: E402


DIMENSION_LABELS: Mapping[PreferenceDimension, str] = {
    PreferenceDimension.TIME: "时间",
    PreferenceDimension.COST: "费用",
    PreferenceDimension.WALKING_DISTANCE: "步行距离",
    PreferenceDimension.TRANSFERS: "换乘次数",
}

ROUTE_PAIR_COUNT = 96
ROUTE_RANDOM_SEED = 20260805
MINIMUM_TARGET_SCORE_GAP = 0.015
PROGRESS_CHECKPOINTS = (12, 24, 48)
DEFAULT_CHOICE_DISPLAY_LIMIT = 5


@dataclass(frozen=True, slots=True)
class SyntheticProfile:
    """用于自动产生路线选择的已知目标画像。"""

    name: str
    sensitivities: Mapping[PreferenceDimension, float]

    @property
    def dominant_dimension(self) -> PreferenceDimension:
        return max(PREFERENCE_DIMENSIONS, key=self.sensitivities.__getitem__)


@dataclass(frozen=True, slots=True)
class RoutePair:
    """等待目标画像自动选择的两条候选路线。"""

    route_a: RouteAttributes
    route_b: RouteAttributes


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    """一类目标画像的自动验证结果。"""

    profile: SyntheticProfile
    comparisons: tuple[PairwisePreference, ...]
    result: PreferenceLearningResult
    mean_absolute_error: float
    mean_choice_probability: float
    checkpoint_dominants: tuple[PreferenceDimension, ...]


def _dominant_profile(
    name: str,
    dominant: PreferenceDimension,
) -> SyntheticProfile:
    """构造主偏好占70%、其他偏好各占10%的清晰画像。"""

    sensitivities = {
        dimension: 0.70 if dimension is dominant else 0.10
        for dimension in PREFERENCE_DIMENSIONS
    }
    return SyntheticProfile(name=name, sensitivities=sensitivities)


TARGET_PROFILES = (
    _dominant_profile("时间优先用户", PreferenceDimension.TIME),
    _dominant_profile("费用优先用户", PreferenceDimension.COST),
    _dominant_profile("少步行优先用户", PreferenceDimension.WALKING_DISTANCE),
    _dominant_profile("少换乘优先用户", PreferenceDimension.TRANSFERS),
)


def build_route_pair_list(count: int = ROUTE_PAIR_COUNT) -> list[RoutePair]:
    """生成确定且可复现的候选路线对列表。

    所有数值都位于当前归一化尺度内，避免截断让不同路线失去差异。固定随机
    种子使每次运行得到完全相同的路线列表和验证结果。
    """

    random = Random(ROUTE_RANDOM_SEED)
    route_pairs: list[RoutePair] = []
    for index in range(1, count + 1):

        def route(suffix: str) -> RouteAttributes:
            return RouteAttributes(
                route_id=f"auto-{index:03d}-{suffix}",
                total_time_minutes=round(random.uniform(15.0, 150.0), 1),
                total_cost=round(random.uniform(2.0, 90.0), 1),
                walking_distance_meters=round(random.uniform(100.0, 2700.0)),
                transfer_count=random.randint(0, 4),
            )

        route_pairs.append(RoutePair(route("a"), route("b")))
    return route_pairs


def _target_route_score(
    profile: SyntheticProfile,
    route: RouteAttributes,
) -> float:
    """用已知目标画像计算路线代价，分数越低越符合该画像。"""

    return sum(
        profile.sensitivities[dimension]
        * min(
            route.value_for(dimension) / NORMALIZATION_SCALES[dimension],
            1.0,
        )
        for dimension in PREFERENCE_DIMENSIONS
    )


def build_automated_choice_list(
    profile: SyntheticProfile,
    route_pairs: Sequence[RoutePair],
) -> list[PairwisePreference]:
    """把路线对自动转换为“已选路线优于未选路线”的偏好列表。"""

    comparisons: list[PairwisePreference] = []
    for pair in route_pairs:
        score_a = _target_route_score(profile, pair.route_a)
        score_b = _target_route_score(profile, pair.route_b)
        if abs(score_a - score_b) < MINIMUM_TARGET_SCORE_GAP:
            # 目标画像对两条路线几乎无差异时跳过，避免人为规定模糊选择。
            continue

        chosen, rejected = (
            (pair.route_a, pair.route_b)
            if score_a < score_b
            else (pair.route_b, pair.route_a)
        )
        comparisons.append(PairwisePreference(chosen=chosen, rejected=rejected))
    return comparisons


def _mean_absolute_error(
    expected: Mapping[PreferenceDimension, float],
    actual: Mapping[PreferenceDimension, float],
) -> float:
    return sum(
        abs(expected[dimension] - actual[dimension])
        for dimension in PREFERENCE_DIMENSIONS
    ) / len(PREFERENCE_DIMENSIONS)


def validate_profile(
    profile: SyntheticProfile,
    route_pairs: Sequence[RoutePair],
) -> ValidationSummary:
    """自动学习一类画像，并检查主偏好、误差、概率和增量结果。"""

    comparisons = build_automated_choice_list(profile, route_pairs)
    learner = PairwisePreferenceWeightLearner()
    result = learner.fit(comparisons)
    learned_dominant = max(PREFERENCE_DIMENSIONS, key=result.weights.__getitem__)
    mean_probability = sum(result.choice_probabilities) / len(
        result.choice_probabilities
    )
    error = _mean_absolute_error(profile.sensitivities, result.weights)

    checkpoint_dominants = tuple(
        max(
            PREFERENCE_DIMENSIONS,
            key=learner.fit(comparisons[:count]).weights.__getitem__,
        )
        for count in PROGRESS_CHECKPOINTS
    )

    assert len(comparisons) >= 80, "有效路线选择不足，无法形成稳定验证"
    assert result.evidence_count == len(comparisons), "路线选择没有全部进入学习器"
    assert result.converged, "个人画像数值优化没有完全收敛"
    assert learned_dominant is profile.dominant_dimension, (
        f"{profile.name}的主偏好识别错误："
        f"预期{DIMENSION_LABELS[profile.dominant_dimension]}，"
        f"实际{DIMENSION_LABELS[learned_dominant]}"
    )
    assert result.weights[learned_dominant] >= 0.50, "主偏好权重不够明显"
    assert error <= 0.10, f"学习画像与目标画像平均误差过大：{error:.2%}"
    assert mean_probability >= 0.70, "模型对自动选择的平均后验概率过低"
    assert all(
        dimension is profile.dominant_dimension
        for dimension in checkpoint_dominants
    ), "增量画像在部分检查点没有识别出预设主偏好"

    return ValidationSummary(
        profile=profile,
        comparisons=tuple(comparisons),
        result=result,
        mean_absolute_error=error,
        mean_choice_probability=mean_probability,
        checkpoint_dominants=checkpoint_dominants,
    )


def _format_weights(weights: Mapping[PreferenceDimension, float]) -> str:
    return "，".join(
        f"{DIMENSION_LABELS[dimension]} {weights[dimension]:.1%}"
        for dimension in PREFERENCE_DIMENSIONS
    )


def _format_route(route: RouteAttributes) -> str:
    """展示进入画像学习器的完整四维路线属性。"""

    return (
        f"时间 {route.total_time_minutes:g} 分钟｜"
        f"费用 {route.total_cost:g} 元｜"
        f"步行 {route.walking_distance_meters:g} 米｜"
        f"换乘 {route.transfer_count} 次"
    )


def _print_choice_details(
    summary: ValidationSummary,
    limit: int | None,
) -> None:
    """展示自动选择的路线属性、目标分数和选择依据。"""

    comparisons = (
        summary.comparisons
        if limit is None
        else summary.comparisons[:limit]
    )
    print(
        f"  自动路线选择明细（显示 {len(comparisons)}/"
        f"{len(summary.comparisons)} 条）："
    )
    for index, comparison in enumerate(comparisons, start=1):
        chosen_score = _target_route_score(summary.profile, comparison.chosen)
        rejected_score = _target_route_score(summary.profile, comparison.rejected)
        print(f"    [{index}] 选择 {comparison.chosen.route_id}")
        print(f"        {_format_route(comparison.chosen)}")
        print(f"        自动选择评分：{chosen_score:.4f}")
        print(f"        未选择 {comparison.rejected.route_id}")
        print(f"        {_format_route(comparison.rejected)}")
        print(f"        自动选择评分：{rejected_score:.4f}")
        print(
            f"        选择依据：已选路线加权代价低 "
            f"{rejected_score - chosen_score:.4f}"
        )

    omitted = len(summary.comparisons) - len(comparisons)
    if omitted:
        print(
            f"    其余 {omitted} 条已省略；使用 --show-all 可查看全部路线选择。"
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="输出四类目标画像的全部自动路线选择，而不只显示前五条",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    route_pairs = build_route_pair_list()
    print("FAVOUR 四维个人画像自动化验证")
    print(f"固定生成 {len(route_pairs)} 组候选路线对，不需要任何终端选择。")

    summaries = tuple(
        validate_profile(profile, route_pairs) for profile in TARGET_PROFILES
    )
    for summary in summaries:
        learned_dominant = max(
            PREFERENCE_DIMENSIONS,
            key=summary.result.weights.__getitem__,
        )
        checkpoints = " → ".join(
            DIMENSION_LABELS[dimension]
            for dimension in summary.checkpoint_dominants
        )
        print(f"\n{summary.profile.name}：验证通过")
        print(f"  学习画像：{_format_weights(summary.result.weights)}")
        print(
            f"  主偏好：{DIMENSION_LABELS[learned_dominant]}｜"
            f"有效选择：{len(summary.comparisons)} 条｜"
            f"平均误差：{summary.mean_absolute_error:.2%}｜"
            f"平均后验概率：{summary.mean_choice_probability:.2%}"
        )
        print(f"  12/24/48条证据下的主偏好：{checkpoints}")
        _print_choice_details(
            summary,
            limit=None if args.show_all else DEFAULT_CHOICE_DISPLAY_LIMIT,
        )

    print("\n全部自动化画像验证通过：路线选择列表已进入正式学习流程。")


if __name__ == "__main__":
    main()
