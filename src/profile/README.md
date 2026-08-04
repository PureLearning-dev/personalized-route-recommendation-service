# FAVOUR 四维长期偏好画像模块

本模块根据用户对路线的 A/B 选择，学习时间、费用、步行距离和换乘次数四项
长期代价敏感度。核心实现包括 Gaussian 先验、Bradley-Terry/Logit 似然、
Laplace 后验、增量 Bayes 更新、MPP 聚合和考虑后验协方差的选择预测。

模型内部系数不要求总和为 1；界面上的四维百分比只是相对展示结果。完整的
公式映射、每个 Python 文件职责、文件交互和流程图见
[FAVOUR实现与模块交互说明](FAVOUR实现与模块交互说明.md)。

## 运行验证

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q src tests examples
python3 examples/interactive_profile_demo.py
```

当前实现是四维缩减特征版本，不包含 FAVOUR 论文的全部 59 个路线及情境特征，
也不负责候选路线生成和最终路线排序。
