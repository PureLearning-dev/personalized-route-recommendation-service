"""个性化路线推荐模块的领域异常。"""


class RecommendationValidationError(ValueError):
    """推荐输入不完整或不满足业务约束。"""

