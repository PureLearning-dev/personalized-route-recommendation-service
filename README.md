# 个性化推荐服务

本项目实现“画像构建”和“路线推荐”两个部分的内容。

通过构建用户的画像，再根据用户的画像去候选路径中选择最推荐的路线，这是项目的核心任务。

## 如何构建用户画像

用户的画像被构建为 4 维，包括：时间、费用、步行、换乘，最终得到的画像是维度系数和各自的百分比值，含义为用户重视该纬度的程度。


## 如何选择最推荐的路线

## 运行验证

创造合适的情况进行验证。



```bash
python3 -m unittest discover -s tests -v
python3 examples/interactive_profile_demo.py
python3 examples/preset_profile_demo.py
python3 examples/personalized_route_ranking_demo.py
python3 examples/jnd_enhanced_route_ranking_demo.py
```
