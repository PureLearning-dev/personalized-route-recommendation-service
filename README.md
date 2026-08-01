# 个性化推荐服务

本项目用于研究多模式出行中的个性化推荐。

当前代码重点验证第一版长期基础偏好学习：用户比较6组设计好的路线，系统
根据选择反推出其对时间、费用、步行距离和换乘次数的偏好权重。

- 设计与使用说明：[src/profile/README.md](src/profile/README.md)
- 交互式体验：[examples/interactive_profile_demo.py](examples/interactive_profile_demo.py)

运行验证：

```bash
python3 -m unittest discover -s tests -v
python3 examples/interactive_profile_demo.py
```
