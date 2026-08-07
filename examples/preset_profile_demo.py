"""展示多种预设画像如何作为先验接入正式个人画像学习流程。"""

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


DIMENSION_LABELS: Mapping[PreferenceDimension, str] = {
    PreferenceDimension.TIME: "时间",
    PreferenceDimension.COST: "费用",
    PreferenceDimension.WALKING_DISTANCE: "步行距离",
    PreferenceDimension.TRANSFERS: "换乘次数",
}

PRESET_LABELS: Mapping[PreferencePreset, str] = {
    PreferencePreset.BALANCED: "均衡型",
    PreferencePreset.TIME_PRIORITY: "时间优先型",
    PreferencePreset.COST_PRIORITY: "费用优先型",
    PreferencePreset.LOW_WALKING: "少步行型",
    PreferencePreset.LOW_TRANSFERS: "少换乘型",
}


def _format_weights(weights: Mapping[PreferenceDimension, float]) -> str:
    return "｜".join(
        f"{DIMENSION_LABELS[dimension]} {weights[dimension]:.1%}"
        for dimension in PREFERENCE_DIMENSIONS
    )


def _cost_sensitive_choices(count: int) -> tuple[PairwisePreference, ...]:
    """生成用户为节省费用而接受更长时间的明确选择证据。"""

    return tuple(
        PairwisePreference(
            chosen=RouteAttributes(f"cheap-{index}", 60, 0, 300, 1),
            rejected=RouteAttributes(f"fast-{index}", 30, 100, 300, 1),
        )
        for index in range(1, count + 1)
    )


def main() -> None:
    learner = PairwisePreferenceWeightLearner()
    print("预设用户画像已接入个人画像学习流程")
    print("\n没有个人选择时，各类用户不再只能返回四项25%：")
    for preset in PreferencePreset:
        result = learner.fit((), preference_preset=preset)
        print(f"  {PRESET_LABELS[preset]}：{_format_weights(result.weights)}")

    initial = learner.fit(
        (),
        preference_preset=PreferencePreset.TIME_PRIORITY,
    )
    choices = _cost_sensitive_choices(24)
    updated = learner.fit(
        choices,
        preference_preset=PreferencePreset.TIME_PRIORITY,
    )
    print("\n预设画像不会固定最终结果，用户真实选择会继续更新画像：")
    print(f"  初始时间优先画像：{_format_weights(initial.weights)}")
    print(f"  加入 {len(choices)} 条费用优先选择后：{_format_weights(updated.weights)}")


if __name__ == "__main__":
    main()
