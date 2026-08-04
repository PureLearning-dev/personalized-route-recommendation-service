"""FAVOUR Laplace 推断使用的无业务依赖数值工具。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import exp, isfinite, log1p

from .exceptions import ProfileNumericalError, ProfileValidationError
from .models import Matrix


Vector = tuple[float, ...]


def sigmoid(value: float) -> float:
    """数值稳定的 Sigmoid。"""

    if value >= 0:
        inverse = exp(-value)
        return 1.0 / (1.0 + inverse)
    direct = exp(value)
    return direct / (1.0 + direct)


def softplus(value: float) -> float:
    """数值稳定地计算 log(1 + exp(value))。"""

    if value > 0:
        return value + log1p(exp(-value))
    return log1p(exp(value))


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def matrix_vector_product(matrix: Matrix, vector: Sequence[float]) -> Vector:
    return tuple(dot(row, vector) for row in matrix)


def quadratic_form(vector: Sequence[float], matrix: Matrix) -> float:
    return dot(vector, matrix_vector_product(matrix, vector))


def add_diagonal(matrix: Matrix, value: float) -> Matrix:
    return tuple(
        tuple(
            current + value if row_index == column_index else current
            for column_index, current in enumerate(row)
        )
        for row_index, row in enumerate(matrix)
    )


def symmetrize(matrix: Matrix) -> Matrix:
    size = len(matrix)
    return tuple(
        tuple((matrix[row][column] + matrix[column][row]) / 2.0 for column in range(size))
        for row in range(size)
    )


def solve_linear_system(matrix: Matrix, values: Sequence[float]) -> Vector:
    """使用带部分主元选择的高斯消元求解小型线性方程。"""

    size = len(matrix)
    if size == 0 or len(values) != size or any(len(row) != size for row in matrix):
        raise ProfileValidationError("线性方程的矩阵和向量维度不匹配")

    augmented = [
        [float(value) for value in row] + [float(values[row_index])]
        for row_index, row in enumerate(matrix)
    ]

    for column in range(size):
        pivot_row = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        pivot = augmented[pivot_row][column]
        if abs(pivot) <= 1e-14:
            raise ProfileNumericalError("矩阵接近奇异，无法稳定求解")
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
        raise ProfileNumericalError("线性方程产生了非有限结果")
    return tuple(solution)


def inverse_matrix(matrix: Matrix, initial_jitter: float = 0.0) -> Matrix:
    """求小型对称矩阵的逆；必要时逐步增加对角抖动。"""

    size = len(matrix)
    jitters = (initial_jitter, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4)
    last_error: ProfileNumericalError | None = None
    for jitter in dict.fromkeys(jitters):
        try:
            candidate = add_diagonal(matrix, jitter) if jitter else matrix
            columns = []
            for column in range(size):
                basis = tuple(1.0 if row == column else 0.0 for row in range(size))
                columns.append(solve_linear_system(candidate, basis))
            inverse = tuple(
                tuple(columns[column][row] for column in range(size))
                for row in range(size)
            )
            return symmetrize(inverse)
        except ProfileNumericalError as error:
            last_error = error
    raise ProfileNumericalError("矩阵求逆失败") from last_error


@dataclass(frozen=True, slots=True)
class ObjectiveEvaluation:
    value: float
    gradient: Vector
    hessian: Matrix


@dataclass(frozen=True, slots=True)
class NewtonOptimizerConfig:
    max_iterations: int = 200
    tolerance: float = 1e-9
    initial_jitter: float = 1e-10
    line_search_shrink: float = 0.5
    armijo_constant: float = 1e-4
    minimum_step_size: float = 1e-10

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ProfileValidationError("最大迭代次数必须为正数")
        if not 0 < self.line_search_shrink < 1:
            raise ProfileValidationError("线搜索缩减率必须位于 (0, 1)")
        if not 0 < self.armijo_constant < 1:
            raise ProfileValidationError("Armijo 常数必须位于 (0, 1)")
        for field_name in ("tolerance", "initial_jitter", "minimum_step_size"):
            value = float(getattr(self, field_name))
            if not isfinite(value) or value <= 0:
                raise ProfileValidationError(f"{field_name} 必须是有限的正数")


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    point: Vector
    evaluation: ObjectiveEvaluation
    converged: bool
    iterations: int


class BoxConstrainedNewtonOptimizer:
    """解析 Hessian、箱型投影和回溯线搜索组成的阻尼 Newton 求解器。"""

    def __init__(self, config: NewtonOptimizerConfig | None = None) -> None:
        self._config = config or NewtonOptimizerConfig()

    @staticmethod
    def _project(
        point: Sequence[float],
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
    ) -> Vector:
        return tuple(
            min(max(value, lower), upper)
            for value, lower, upper in zip(
                point,
                lower_bounds,
                upper_bounds,
                strict=True,
            )
        )

    def _projected_gradient_norm(
        self,
        point: Vector,
        gradient: Vector,
        lower_bounds: Vector,
        upper_bounds: Vector,
    ) -> float:
        active_gradient = []
        for value, component, lower, upper in zip(
            point,
            gradient,
            lower_bounds,
            upper_bounds,
            strict=True,
        ):
            at_lower = value <= lower + self._config.tolerance and component > 0
            at_upper = value >= upper - self._config.tolerance and component < 0
            active_gradient.append(0.0 if at_lower or at_upper else component)
        return max(abs(component) for component in active_gradient)

    def _search(
        self,
        objective: Callable[[Vector], ObjectiveEvaluation],
        current: Vector,
        current_evaluation: ObjectiveEvaluation,
        direction: Vector,
        lower_bounds: Vector,
        upper_bounds: Vector,
    ) -> tuple[Vector, ObjectiveEvaluation] | None:
        step_size = 1.0
        while step_size >= self._config.minimum_step_size:
            candidate = self._project(
                tuple(
                    value - step_size * component
                    for value, component in zip(current, direction, strict=True)
                ),
                lower_bounds,
                upper_bounds,
            )
            delta = tuple(new - old for new, old in zip(candidate, current, strict=True))
            if max(abs(component) for component in delta) <= self._config.tolerance:
                return candidate, objective(candidate)
            candidate_evaluation = objective(candidate)
            armijo_limit = current_evaluation.value + self._config.armijo_constant * dot(
                current_evaluation.gradient,
                delta,
            )
            if candidate_evaluation.value <= armijo_limit:
                return candidate, candidate_evaluation
            step_size *= self._config.line_search_shrink
        return None

    def optimize(
        self,
        objective: Callable[[Vector], ObjectiveEvaluation],
        initial_point: Sequence[float],
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
    ) -> OptimizationResult:
        lower = tuple(float(value) for value in lower_bounds)
        upper = tuple(float(value) for value in upper_bounds)
        current = self._project(initial_point, lower, upper)
        evaluation = objective(current)

        for iteration in range(1, self._config.max_iterations + 1):
            if self._projected_gradient_norm(
                current,
                evaluation.gradient,
                lower,
                upper,
            ) <= self._config.tolerance:
                return OptimizationResult(current, evaluation, True, iteration - 1)

            try:
                direction = solve_linear_system(
                    add_diagonal(evaluation.hessian, self._config.initial_jitter),
                    evaluation.gradient,
                )
            except ProfileNumericalError:
                direction = evaluation.gradient

            searched = self._search(
                objective,
                current,
                evaluation,
                direction,
                lower,
                upper,
            )
            if searched is None:
                gradient_scale = max(1.0, max(abs(value) for value in evaluation.gradient))
                searched = self._search(
                    objective,
                    current,
                    evaluation,
                    tuple(value / gradient_scale for value in evaluation.gradient),
                    lower,
                    upper,
                )
            if searched is None:
                return OptimizationResult(current, evaluation, False, iteration)

            candidate, candidate_evaluation = searched
            max_change = max(
                abs(new - old) for new, old in zip(candidate, current, strict=True)
            )
            objective_change = abs(evaluation.value - candidate_evaluation.value)
            current, evaluation = candidate, candidate_evaluation
            if max_change <= self._config.tolerance or objective_change <= self._config.tolerance:
                return OptimizationResult(current, evaluation, True, iteration)

        return OptimizationResult(
            current,
            evaluation,
            False,
            self._config.max_iterations,
        )
