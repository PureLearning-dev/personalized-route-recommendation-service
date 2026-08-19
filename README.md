# 个性化推荐服务

本项目用于研究多模式出行中的个性化推荐。

当前代码已经覆盖第一版长期基础偏好学习和候选路线推荐：用户比较设计好的
路线后，系统反推出其对时间、费用、步行距离和换乘次数的偏好权重；推荐阶段
先执行硬约束过滤和加权初排，也可以对前N条路线使用JND进一步精排。

- 设计与使用说明：[src/profile/README.md](src/profile/README.md)
- 交互式体验：[examples/interactive_profile_demo.py](examples/interactive_profile_demo.py)
- 预设画像演示：[examples/preset_profile_demo.py](examples/preset_profile_demo.py)
- 候选路线个性化排序：[src/recommendation/README.md](src/recommendation/README.md)
- 路线排序演示：[examples/personalized_route_ranking_demo.py](examples/personalized_route_ranking_demo.py)
- 加权与JND两阶段排序演示：[examples/jnd_enhanced_route_ranking_demo.py](examples/jnd_enhanced_route_ranking_demo.py)

运行验证：

```bash
python3 -m unittest discover -s tests -v
python3 examples/interactive_profile_demo.py
python3 examples/preset_profile_demo.py
python3 examples/personalized_route_ranking_demo.py
python3 examples/jnd_enhanced_route_ranking_demo.py
```
