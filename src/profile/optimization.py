"""论文所述 box-bounded trust-region 优化及四维矩阵运算。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import exp, isfinite, log1p, sqrt
from random import Random

from .exceptions import ProfileNumericalError, ProfileValidationError
from .models import Matrix, Vector


def sigmoid(value: float) -> float:
    if value >= 0:
        inverse = exp(-value)
        return 1.0 / (1.0 + inverse)
    direct = exp(value)
    return direct / (1.0 + direct)


def softplus(value: float) -> float:
    if value > 0:
        return value + log1p(exp(-value))
    return log1p(exp(value))


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def norm(vector: Sequence[float]) -> float:
    return sqrt(dot(vector, vector))


def matrix_vector_product(matrix: Matrix, vector: Sequence[float]) -> Vector:
    return tuple(dot(row, vector) for row in matrix)


def quadratic_form(vector: Sequence[float], matrix: Matrix) -> float:
    return dot(vector, matrix_vector_product(matrix, vector))


def add_diagonal(matrix: Matrix, value: float) -> Matrix:
    return tuple(
        tuple(
            current + value if row == column else current
            for column, current in enumerate(values)
        )
        for row, values in enumerate(matrix)
    )


def symmetrize(matrix: Matrix) -> Matrix:
    size = len(matrix)
    return tuple(
        tuple((matrix[row][column] + matrix[column][row]) / 2.0 for column in range(size))
        for row in range(size)
    )


def solve_linear_system(matrix: Matrix, values: Sequence[float]) -> Vector:
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix) or len(values) != size:
        raise ProfileValidationError("线性方程的矩阵和向量维度必须一致")
    augmented = [
        [float(value) for value in row] + [float(values[index])]
        for index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot_row = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot_row][column]) <= 1e-14:
            raise ProfileNumericalError("矩阵接近奇异，无法求解")
        augmented[column], augmented[pivot_row] = augmented[pivot_row], augmented[column]
        for row in range(column + 1, size):
            factor = augmented[row][column] / augmented[column][column]
            for index in range(column, size + 1):
                augmented[row][index] -= factor * augmented[column][index]

    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        residual = augmented[row][size] - sum(
            augmented[row][column] * solution[column]
            for column in range(row + 1, size)
        )
        solution[row] = residual / augmented[row][row]
    if any(not isfinite(value) for value in solution):
        raise ProfileNumericalError("线性方程产生非有限结果")
    return tuple(solution)


def inverse_matrix(matrix: Matrix) -> Matrix:
    size = len(matrix)
    columns = [
        solve_linear_system(
            matrix,
            tuple(1.0 if row == column else 0.0 for row in range(size)),
        )
        for column in range(size)
    ]
    return symmetrize(
        tuple(
            tuple(columns[column][row] for column in range(size))
            for row in range(size)
        )
    )


def determinant(matrix: Matrix) -> float:
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ProfileValidationError("行列式计算需要非空方阵")
    values = [list(row) for row in matrix]
    result = 1.0
    for column in range(len(values)):
        pivot_row = max(range(column, len(values)), key=lambda row: abs(values[row][column]))
        if abs(values[pivot_row][column]) <= 1e-14:
            raise ProfileNumericalError("协方差矩阵行列式为零")
        if pivot_row != column:
            values[column], values[pivot_row] = values[pivot_row], values[column]
            result *= -1.0
        pivot = values[column][column]
        result *= pivot
        for row in range(column + 1, len(values)):
            factor = values[row][column] / pivot
            for index in range(column + 1, len(values)):
                values[row][index] -= factor * values[column][index]
    return result


@dataclass(frozen=True, slots=True)
class ObjectiveEvaluation:
    value: float
    gradient: Vector
    hessian: Matrix


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    point: Vector
    evaluation: ObjectiveEvaluation
    converged: bool


class BoxBoundedTrustRegionOptimizer:
    """按论文设置执行五次随机起点的箱约束trust-region优化。"""

    _RUNS = 5
    _SEED = 1
    _MAX_ITERATIONS = 200
    # 四维纯 Python 求解器使用该绝对容差，避免在浮点噪声附近空转。
    _GRADIENT_TOLERANCE = 1e-7

    @staticmethod
    def _project(point: Sequence[float], lower: Vector, upper: Vector) -> Vector:
        return tuple(
            min(max(value, low), high)
            for value, low, high in zip(point, lower, upper, strict=True)
        )

    @staticmethod
    def _is_blocked(
        value: float,
        gradient: float,
        lower: float,
        upper: float,
    ) -> bool:
        """判断负梯度方向是否会把当前分量推出箱约束。"""

        return (value <= lower + 1e-8 and gradient > 0.0) or (
            value >= upper - 1e-8 and gradient < 0.0
        )

    @classmethod
    def _projected_gradient(
        cls,
        point: Vector,
        gradient: Vector,
        lower: Vector,
        upper: Vector,
    ) -> Vector:
        projected = []
        for value, component, low, high in zip(
            point,
            gradient,
            lower,
            upper,
            strict=True,
        ):
            blocked = cls._is_blocked(value, component, low, high)
            projected.append(0.0 if blocked else component)
        return tuple(projected)

    @classmethod
    def _projected_gradient_norm(
        cls,
        point: Vector,
        gradient: Vector,
        lower: Vector,
        upper: Vector,
    ) -> float:
        projected = cls._projected_gradient(point, gradient, lower, upper)
        return max(abs(component) for component in projected)

    @classmethod
    def _gradient_candidate(
        cls,
        point: Vector,
        gradient: Vector,
        lower: Vector,
        upper: Vector,
        radius: float,
    ) -> Vector:
        """沿箱约束下的可行负梯度方向生成备用候选点。"""

        direction = tuple(
            -component
            for component in cls._projected_gradient(
                point,
                gradient,
                lower,
                upper,
            )
        )
        direction_norm = norm(direction)
        if direction_norm > radius:
            direction = tuple(
                radius * value / direction_norm for value in direction
            )
        return cls._project(
            tuple(
                value + change
                for value, change in zip(point, direction, strict=True)
            ),
            lower,
            upper,
        )

    def _run(
        self,
        objective: Callable[[Vector], ObjectiveEvaluation],
        start: Vector,
        lower: Vector,
        upper: Vector,
    ) -> OptimizationResult:
        current = self._project(start, lower, upper)
        evaluation = objective(current)
        radius = 1.0
        maximum_radius = max(1.0, norm(tuple(high - low for low, high in zip(lower, upper))))

        for _ in range(self._MAX_ITERATIONS):
            if self._projected_gradient_norm(
                current,
                evaluation.gradient,
                lower,
                upper,
            ) <= self._GRADIENT_TOLERANCE:
                return OptimizationResult(current, evaluation, True)

            free_indices = tuple(
                index
                for index, (value, gradient, low, high) in enumerate(
                    zip(
                        current,
                        evaluation.gradient,
                        lower,
                        upper,
                        strict=True,
                    )
                )
                if not self._is_blocked(value, gradient, low, high)
            )
            # 只在自由变量子空间求 Newton 步。若把已抵住边界的变量继续放入
            # 线性方程，其耦合项会扭曲其他维度的方向并造成边界附近来回振荡。
            reduced_hessian = tuple(
                tuple(evaluation.hessian[row][column] for column in free_indices)
                for row in free_indices
            )
            reduced_step = solve_linear_system(
                add_diagonal(reduced_hessian, 1e-10),
                tuple(-evaluation.gradient[index] for index in free_indices),
            )
            free_step = dict(zip(free_indices, reduced_step, strict=True))
            step = tuple(free_step.get(index, 0.0) for index in range(len(current)))
            step_norm = norm(step)
            if step_norm > radius:
                step = tuple(radius * value / step_norm for value in step)

            candidate = self._project(
                tuple(value + change for value, change in zip(current, step, strict=True)),
                lower,
                upper,
            )
            actual_step = tuple(
                new - old for new, old in zip(candidate, current, strict=True)
            )
            if norm(actual_step) <= 1e-10:
                # Newton 方向可能被箱边界完全截断；此时改用可行的负梯度方向，
                # 避免把仍可下降的边界点误报为收敛或失败。
                candidate = self._gradient_candidate(
                    current,
                    evaluation.gradient,
                    lower,
                    upper,
                    radius,
                )
                actual_step = tuple(
                    new - old
                    for new, old in zip(candidate, current, strict=True)
                )
                if norm(actual_step) <= 1e-10:
                    return OptimizationResult(current, evaluation, False)

            predicted_reduction = -(
                dot(evaluation.gradient, actual_step)
                + 0.5 * quadratic_form(actual_step, evaluation.hessian)
            )
            if predicted_reduction <= 0:
                # Newton 模型在活动边界上可能给出非下降步，使用投影梯度重新尝试。
                candidate = self._gradient_candidate(
                    current,
                    evaluation.gradient,
                    lower,
                    upper,
                    radius,
                )
                actual_step = tuple(
                    new - old
                    for new, old in zip(candidate, current, strict=True)
                )
                predicted_reduction = -(
                    dot(evaluation.gradient, actual_step)
                    + 0.5 * quadratic_form(actual_step, evaluation.hessian)
                )
                if predicted_reduction <= 0 or norm(actual_step) <= 1e-10:
                    radius *= 0.25
                    continue

            candidate_evaluation = objective(candidate)
            actual_reduction = evaluation.value - candidate_evaluation.value
            ratio = actual_reduction / predicted_reduction

            if ratio < 0.25:
                radius *= 0.25
            elif ratio > 0.75 and norm(actual_step) >= 0.9 * radius:
                radius = min(2.0 * radius, maximum_radius)

            if ratio > 0.1:
                current, evaluation = candidate, candidate_evaluation

        return OptimizationResult(current, evaluation, False)

    def optimize(
        self,
        objective: Callable[[Vector], ObjectiveEvaluation],
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
    ) -> OptimizationResult:
        lower = tuple(float(value) for value in lower_bounds)
        upper = tuple(float(value) for value in upper_bounds)
        if not lower or len(lower) != len(upper):
            raise ProfileValidationError("优化器上下界必须是同长度的非空向量")
        if any(
            not isfinite(low) or not isfinite(high) or low > high
            for low, high in zip(lower, upper, strict=True)
        ):
            raise ProfileValidationError("优化器上下界必须是有限数且下界不大于上界")

        random = Random(self._SEED)
        starts = tuple(
            tuple(random.uniform(low, high) for low, high in zip(lower, upper, strict=True))
            for _ in range(self._RUNS)
        )
        results: list[OptimizationResult] = []
        last_error: ProfileNumericalError | None = None
        for start in starts:
            try:
                results.append(self._run(objective, start, lower, upper))
            except ProfileNumericalError as error:
                # 单个随机起点失败时仍允许其他起点完成论文规定的多起点搜索。
                last_error = error

        converged = [result for result in results if result.converged]
        if not converged:
            raise ProfileNumericalError("所有随机起点均未得到收敛结果") from last_error
        return min(converged, key=lambda result: result.evaluation.value)
