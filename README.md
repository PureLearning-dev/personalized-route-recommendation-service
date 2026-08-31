# 个性化路线推荐服务

项目把用户对路线的真实选择学习为时间、费用、步行距离、换乘次数四维画像，并使用当前画像对候选路线进行排序。

当前阶段坚持最小实现，只保留两个需要持久化的核心对象：用户和用户当前画像。

## 数据库

PostgreSQL 只有两张业务表：

- `users`：用户编号和画像初始化方式。
- `user_profiles`：用户当前的四维系数、协方差和有效学习次数。

四维权重和百分比由系数实时计算，不重复保存。每次选择学习后直接更新 `user_profiles` 的同一行，不保存画像历史、推荐会话或路线快照。

## 核心流程

```text
创建用户和初始画像
  → 保存到 users、user_profiles
  → 传入候选路线，即时计算推荐结果（不保存路线）
  → 传入选中路线和对比路线
  → 基于当前 Gaussian 后验继续学习
  → 更新 user_profiles 的同一行
```

## Docker 启动

```bash
docker compose up --build -d
docker compose ps
```

启动后访问：

- API 文档：http://127.0.0.1:8000/docs
- 就绪检查：http://127.0.0.1:8000/health/ready

查看真实数据库表：

```bash
docker compose exec postgres sh -lc 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\dt"'
```

## API

完整的请求参数、返回字段、调用示例和联调顺序见根目录的 [API接口使用说明.md](API接口使用说明.md)。

- `POST /v1/users`：创建用户和初始画像。
- `GET /v1/users`：查询全部用户及其当前画像。
- `GET /v1/users/{user_id}/profile`：查看当前画像。
- `DELETE /v1/users/{user_id}`：删除用户及其画像。
- `POST /v1/recommendations`：根据当前画像即时排序候选路线。
- `POST /v1/users/{user_id}/choices`：提交一组选中/对比路线并同步更新画像。

当前没有 API Key、幂等键、异步任务、画像历史和撤回功能。

## 验证

```bash
uv sync --extra dev
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_api_effect.py
```

集成测试使用独立数据库 `personalized_recommendations_core_test`。第一次运行测试前创建：

```bash
docker compose exec -T postgres createdb -U personalized personalized_recommendations_core_test
```

验证脚本会创建默认画像、即时推荐路线、提交一次选择，然后确认更新后的画像已经保存。

## 当前边界

- 推荐接口只排序调用方传入的候选路线，不搜索交通网络。
- 一次反馈直接传入选中路线和一条对比路线。
- 不保存路线和选择历史，因此暂不支持撤回或从完整历史重算画像。
- 当前画像包含继续进行逐题贝叶斯更新所需的四维系数和协方差矩阵。
