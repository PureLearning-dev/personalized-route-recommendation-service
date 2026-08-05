# FAVOUR 四维长期偏好画像

本模块按 FAVOUR 论文流程，由用户对路线的成对选择学习时间、费用、
步行距离和换乘次数四项长期偏好。实际计算包含：

- 图 2 的 MPP 初始化与精炼；
- 公式（4）（5）的二元 Logit 选择模型；
- 公式（6）的逐题贝叶斯更新与 Laplace 后验；
- 公式（7）的群体 MPP 聚合；
- 公式（9）的后验选择概率。

论文中没有给出可直接复制的人口学分类规则，因此本模块不虚构群体
匹配算法；调用方在完成群体匹配后，将该群体历史选择传入
`group_histories`。论文实验的 59 维路线及情境特征在此缩减为任务所需的
四项代价特征。

公式、代码对应、文件职责和数据流程见
[FAVOUR实现与模块交互说明](FAVOUR实现与模块交互说明.md)。

## 运行验证

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python3 examples/interactive_profile_demo.py
```
