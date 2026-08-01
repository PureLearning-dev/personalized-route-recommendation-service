"""偏好权重学习模块对外暴露的异常类型。"""


class ProfileError(Exception):
    """画像权重学习相关异常的统一基类。"""


class ProfileValidationError(ProfileError, ValueError):
    """路线、权重或学习配置不符合领域约束。"""
