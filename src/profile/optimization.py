"""偏好学习使用的小型数值优化工具。

这些函数与交通业务无关，单独放置可以让学习器保持聚焦，也便于针对数值
稳定性进行独立测试。项目第一版只有四个权重，无需引入大型优化依赖。
"""

from __future__ import annotations

from math import exp, log1p
from typing import Sequence


def sigmoid(value: float) -> float:
    """数值稳定的 Sigmoid，避免较大绝对值导致 ``exp`` 溢出。"""

    if value >= 0:
        inverse = exp(-value)
        return 1.0 / (1.0 + inverse)
    direct = exp(value)
    return direct / (1.0 + direct)


def softplus(value: float) -> float:
    """数值稳定地计算 ``log(1 + exp(value))``。"""

    if value > 0:
        return value + log1p(exp(-value))
    return log1p(exp(value))


def project_to_probability_simplex(values: Sequence[float]) -> list[float]:
    """把向量投影到“非负且总和为 1”的概率单纯形。

    偏好权重每次梯度更新后可能暂时出现负数或总和不为 1。本函数使用排序
    阈值算法寻找欧氏距离最近的合法权重向量，从根本上保证画像可解释。
    """

    if not values:
        raise ValueError("待投影向量不能为空")

    ordered = sorted((float(value) for value in values), reverse=True)
    cumulative = 0.0
    active_count = 0
    threshold = 0.0

    for index, value in enumerate(ordered, start=1):
        cumulative += value
        candidate_threshold = (cumulative - 1.0) / index
        if value - candidate_threshold > 0:
            active_count = index
            threshold = candidate_threshold

    # 理论上 active_count 至少为 1；防御分支保证异常输入也不会除以 0。
    if active_count == 0:
        return [1.0 / len(values)] * len(values)

    projected = [max(float(value) - threshold, 0.0) for value in values]
    total = sum(projected)
    return [value / total for value in projected]

