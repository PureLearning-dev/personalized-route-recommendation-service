# 候选路线个性化排序

本模块把 `src.profile` 学到的时间、费用、步行距离和换乘次数四维画像，应用
到上游路径服务提供的候选路线。目前支持两种排序方式：单独使用加权排序，
或者在加权初排后使用JND对前N条路线精排。

1. 本次出行最大可接受值过滤；
2. 四项路线属性归一化；
3. 基于个人画像的加权代价计算；
4. 加权初排并选出前N条路线；
5. 按画像权重确定指标优先级，使用JND精排；
6. 返回最终Top-K、过滤原因和排序过程信息。

## 论文依据

### Route Guidance Ranking Procedures with Human Perception Consideration for Personalized Public Transport Service（2020）

论文：`paper/2026-8-17/1-s2.0-S0968090X20305829-main.pdf`

- 第2.1节公式（1）—（3）：利用属性权重计算路线代价，并通过个人可接受值
  过滤候选路线；对应 `PersonalizedRouteRanker` 和 `RouteConstraints`。
- 第2.2.4节公式（8）及Algorithm 1：各指标使用当前比较路线中的共同最优值
  作为JND参照，再按指标优先级排序，避免简单两两比较产生循环；对应
  `JndEnhancedRouteRanker` 的共同参照值和递归字典序精排。

### Preference-Aware Multimodal Journey Planner（2026）

论文：`paper/个性化推荐服务核心论文/01_Preference_Aware_Multimodal_Journey_Planner_2026.pdf`

- 公式（1）至（4）：使用WSM计算每条候选路线的综合值并完成排序；对应
  `PersonalizedRouteRanker` 的四维加权和。
- 第3.2.2节指出不同量纲指标必须先归一化；本模块复用FAVOUR学习阶段的
  `NORMALIZATION_SCALES`。论文实验使用候选集内min-max归一化，但如果推荐
  阶段改变特征尺度，已学习权重的含义也会改变，因此这里保持训练和推荐一致。
- 第3.3.2节将个性化模块定义为外部路径服务之后的排序/重排序层；本模块同样
  不负责交通网络中的候选路径搜索。

### Route Recommendation Method for Frequent Passengers in Subway Based on Passenger Preference Ranking（2024）

这篇论文进一步根据乘客历史记录调整每个用户的JND阈值。本项目第一版只实现
固定且可配置的四维阈值，暂未实现强化学习或个性化阈值学习；后续积累足够的
真实选择记录后，可以在不改变排序接口的情况下替换阈值来源。

## JND接入流程

1. 先使用原始属性执行硬约束过滤；
2. 对全部可行路线计算个性化代价并加权初排；
3. 只取初排前N条进入JND，且N必须大于或等于K；
4. 按画像权重从高到低确定时间、费用、步行和换乘的比较顺序；
5. 每项指标以Top-N中的最优原始值为参照，差值超过“参照值×JND比例”才算
   可以明显感知；
6. 使用修正后的JND字典序完成精排并截取Top-K。四项都无法区分时，稳定保留
   原来的加权顺序。

这里的“加权初排后再做JND精排”是本项目为了复用现有排序能力采用的组合流程；
两种排序方法和修正JND算法本身来自上述论文，但论文没有规定必须按这个两阶段
顺序组合。

## 使用示例

```python
from src.recommendation import PersonalizedRouteRanker, RouteConstraints

ranking = PersonalizedRouteRanker().rank(
    candidate_routes,
    profile_learning_result,
    constraints=RouteConstraints(max_total_cost=80, max_transfer_count=2),
    top_k=3,
)
```

接入JND精排：

```python
from src.recommendation import JndEnhancedRouteRanker, JndThresholds

result = JndEnhancedRouteRanker().rank(
    candidate_routes,
    profile_learning_result,
    constraints=RouteConstraints(max_total_cost=80, max_transfer_count=2),
    thresholds=JndThresholds(
        time_ratio=0.10,
        cost_ratio=0.10,
        walking_distance_ratio=0.15,
        transfers_ratio=0.0,
    ),
    shortlist_size=5,
    top_k=3,
)
```

`result.weighted_result.ranked_routes` 保存完整加权初排，
`result.ranked_routes` 是JND精排后的Top-K，`attribute_priority`、
`reference_values`、`comparison_steps` 和 `decisive_dimensions` 可用于检查
精排过程。每个 `comparison_steps` 元素会记录本轮路线、指标、共同参考值、
JND允许差值以及范围内外的路线。阈值没有内置默认值，调用方必须明确配置，
避免把未经验证的感知假设写死在算法中。
其中 `decisive_dimensions` 表示一条路线第一次落到JND最优范围之外的指标，
而不是重新计算出的加权贡献。

运行完整示例：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 examples/personalized_route_ranking_demo.py
PYTHONDONTWRITEBYTECODE=1 python3 examples/jnd_enhanced_route_ranking_demo.py
```

第二个示例会依次展示代码中固定的六次路线选择、每次选择后的画像变化、本次
候选路线和过滤结果、每条路线的加权计算，以及JND每一轮的比较组、参考值和
范围内外路线，可作为当前实现的完整过程验证。
