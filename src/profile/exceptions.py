"""偏好学习使用的异常。"""


class ProfileValidationError(ValueError):
    """路线、权重或学习配置不符合领域约束。"""


class ProfileNumericalError(ArithmeticError):
    """偏好后验的数值求解失败。"""
