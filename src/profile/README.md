# 长期偏好权重学习模块

## 当前目标

第一版只完成一个闭环：向用户展示6组具有不同属性取舍的路线，根据用户的
A/B选择，反推出时间、费用、步行距离和换乘次数四项长期基础偏好权重。

当前不包含硬约束、交通方式偏好、本次出行情境、实时环境、数据库持久化、
真实行为增量更新、候选路线生成和最终路线排序。

## 代码结构

```text
src/profile/
├── models.py         # 路线、成对选择、四项维度、先验和诊断数据
├── normalization.py  # 将分钟、元、米和次数转换到统一尺度
├── optimization.py   # Sigmoid、Softplus和合法权重投影
├── learner.py        # 根据多次A/B选择反推四项长期偏好权重
├── exceptions.py     # 输入和配置校验异常
└── __init__.py       # 当前模块的最小公开接口
```

依赖方向保持单向：

```text
交互程序
   ↓
models.py
   ↓
normalization.py + optimization.py
   ↓
learner.py
   ↓
四项长期偏好权重 + 学习诊断
```

## 核心计算

路线属性都表示代价，数值越小越好。对用户选择的路线 `chosen` 和未选择的
路线 `rejected`，先归一化，再构造：

$$
\Delta x=x_{rejected}-x_{chosen}
$$

选择概率为：

$$
P(chosen\succ rejected)=\sigma(w^T\Delta x)
$$

学习器根据所有选择调整 $w$，并始终保证：

$$
w_i\geq0,\qquad \sum_iw_i=1
$$

归一化尺度、学习率、迭代次数和先验强度都通过配置对象提供，未来可以根据
真实数据校准，不需要修改交互流程或学习器接口。

## 运行

手动体验：

```bash
python3 examples/interactive_profile_demo.py
```

自动验证：

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q src tests examples
```
