"""论文所述 box-bounded trust-region 优化及四维矩阵运算。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import exp, isfinite, log1p, sqrt
from random import Random

from .exceptions import ProfileNumericalError
from .models import Matrix


Vector = tuple[float, ...]


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

    @staticmethod
    def _project(point: Sequence[float], lower: Vector, upper: Vector) -> Vector:
        return tuple(
            min(max(value, low), high)
            for value, low, high in zip(point, lower, upper, strict=True)
        )

    @staticmethod
    def _projected_gradient_norm(
        point: Vector,
        gradient: Vector,
        lower: Vector,
        upper: Vector,
    ) -> float:
        projected = []
        for value, component, low, high in zip(
            point,
            gradient,
            lower,
            upper,
            strict=True,
        ):
            blocked = (value <= low + 1e-8 and component > 0) or (
                value >= high - 1e-8 and component < 0
            )
            projected.append(0.0 if blocked else component)
        return max(abs(component) for component in projected)

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

        for _ in range(200):
            if self._projected_gradient_norm(
                current,
                evaluation.gradient,
                lower,
                upper,
            ) <= 1e-8:
                return OptimizationResult(current, evaluation, True)

            step = solve_linear_system(
                add_diagonal(evaluation.hessian, 1e-10),
                tuple(-value for value in evaluation.gradient),
            )
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
                return OptimizationResult(current, evaluation, True)

            predicted_reduction = -(
                dot(evaluation.gradient, actual_step)
                + 0.5 * quadratic_form(actual_step, evaluation.hessian)
            )
            if predicted_reduction <= 0:
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
        random = Random(self._SEED)
        starts = tuple(
            tuple(random.uniform(low, high) for low, high in zip(lower, upper, strict=True))
            for _ in range(self._RUNS)
        )
        results = tuple(
            self._run(objective, start, lower, upper)
            for start in starts
        )
        return min(results, key=lambda result: result.evaluation.value)
