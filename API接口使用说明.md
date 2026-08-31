# 个性化路线推荐 API 接口使用说明

本文档面向需要调用本服务的前端、后端或其他算法服务，说明当前项目对外提供的全部 API、每个请求参数和返回字段的含义，以及推荐的调用方式。

## 1. 服务地址与启动方式

项目使用 FastAPI 提供 HTTP API，使用 PostgreSQL 保存用户和当前画像。

在项目根目录启动：

```bash
docker compose up --build -d
docker compose ps
```

本地默认地址：

- API 根地址：`http://127.0.0.1:8000`
- Swagger 测试页面：`http://127.0.0.1:8000/docs`
- OpenAPI 描述：`http://127.0.0.1:8000/openapi.json`

以下示例统一使用：

```bash
BASE_URL=http://127.0.0.1:8000
```

当前接口没有 API Key，也不需要登录令牌。发送 JSON 时应携带：

```http
Content-Type: application/json
```

## 2. 接口总览

| 方法 | 路径 | 作用 |
|---|---|---|
| `GET` | `/health/live` | 检查 API 进程是否存活 |
| `GET` | `/health/ready` | 检查 API、数据库和数据库版本是否已经可用 |
| `POST` | `/v1/users` | 创建用户并初始化用户画像 |
| `GET` | `/v1/users` | 查询全部用户及其当前画像 |
| `GET` | `/v1/users/{user_id}/profile` | 查询指定用户的当前画像 |
| `POST` | `/v1/users/{user_id}/choices` | 根据一次路线选择学习并更新用户画像 |
| `DELETE` | `/v1/users/{user_id}` | 删除用户及其当前画像 |
| `POST` | `/v1/recommendations` | 使用当前画像对调用方传入的候选路线进行排序 |

## 3. 公共约定

### 3.1 两种用户编号

| 字段 | 谁生成 | 作用 |
|---|---|---|
| `id` / `user_id` | 本服务生成的 UUID | 调用画像、学习、推荐和删除接口时使用 |
| `external_user_id` | 调用方传入 | 保存调用方自己系统中的用户编号，必须全局唯一 |

创建用户后，调用方应保存返回的 `id`。后续接口路径中的 `{user_id}` 和推荐请求中的 `user_id` 都是这个内部 UUID，不是 `external_user_id`。

### 3.2 四个画像维度

| 字段 | 含义 | 路线中的原始单位 |
|---|---|---|
| `time` | 对总出行时间的敏感程度 | 分钟 |
| `cost` | 对总费用的敏感程度 | 当前 Docker 配置为人民币 `CNY` |
| `walking_distance` | 对步行距离的敏感程度 | 米 |
| `transfers` | 对换乘次数的敏感程度 | 次 |

### 3.3 路线对象

推荐接口和画像学习接口都使用相同的路线结构：

```json
{
  "route_id": "route-a",
  "total_time_minutes": 32,
  "total_cost": 15,
  "model_cost_unit": "CNY",
  "walking_distance_meters": 350,
  "transfer_count": 1
}
```

| 属性 | 类型 | 必填 | 取值限制 | 作用 |
|---|---|---:|---|---|
| `route_id` | string | 是 | 长度 1～200；同一次候选集中不能重复 | 路线唯一标识，由上游路径服务生成 |
| `total_time_minutes` | number | 是 | 大于等于 0 | 路线总出行时间，单位为分钟 |
| `total_cost` | number | 是 | 大于等于 0 | 路线总费用 |
| `model_cost_unit` | string | 是 | 当前必须为 `CNY` | 声明 `total_cost` 使用的费用单位；必须与服务配置一致 |
| `walking_distance_meters` | number | 是 | 大于等于 0 | 路线总步行距离，单位为米 |
| `transfer_count` | integer | 是 | 大于等于 0 的整数 | 路线换乘次数 |

当前服务只负责学习和排序，不负责搜索道路或公共交通网络。候选路线及其四项属性必须由调用方或上游路径规划服务提供。

### 3.4 用户画像对象

画像数据包含以下属性：

| 属性 | 类型 | 作用 |
|---|---|---|
| `evidence_count` | integer | 已累计的有效路线选择次数。每成功学习一次通常增加 1 |
| `converged` | boolean | 最近一次画像推断是否收敛。成功返回的更新通常为 `true` |
| `coefficients` | object | 算法实际使用的四维效用系数，包含 `time`、`cost`、`walking_distance`、`transfers` |
| `weights` | object | 由四维系数转换得到的相对权重，四项之和约等于 1 |
| `percentages` | object | `weights × 100`，四项之和约等于 100，方便页面展示 |
| `covariance` | number[][] | 4×4 Gaussian 后验协方差矩阵，保存继续贝叶斯学习所需的不确定性信息 |
| `standard_deviations` | object | 四个维度的后验标准差，用于观察每个系数的不确定程度 |
| `created_at` | datetime | 画像创建时间，ISO 8601 UTC 时间 |
| `updated_at` | datetime | 画像最近更新时间，ISO 8601 UTC 时间 |

`coefficients`、`weights`、`percentages` 和 `standard_deviations` 都包含下面四个键：

```json
{
  "time": 0,
  "cost": 0,
  "walking_distance": 0,
  "transfers": 0
}
```

需要注意：

- `coefficients` 是非正数，当前边界为 `[-20, 0]`。它是算法参数，不是百分比。
- 某个系数越负，表示用户对该项代价越敏感。
- `weights` 才是 0～1 之间的相对占比。
- `percentages` 是最适合直接展示给用户看的百分比。
- `covariance` 的行列顺序固定为：`time`、`cost`、`walking_distance`、`transfers`。
- `standard_deviations` 越小通常表示算法对该系数越确定，它不是偏好大小。

示例：

```json
{
  "evidence_count": 9,
  "converged": true,
  "coefficients": {
    "time": -0.04,
    "cost": -0.51,
    "walking_distance": -1.02,
    "transfers": -0.14
  },
  "weights": {
    "time": 0.023,
    "cost": 0.297,
    "walking_distance": 0.599,
    "transfers": 0.08
  },
  "percentages": {
    "time": 2.3,
    "cost": 29.7,
    "walking_distance": 59.9,
    "transfers": 8.0
  },
  "covariance": [
    [0.98, 0, 0, 0],
    [0, 0.95, 0, 0],
    [0, 0, 0.82, 0],
    [0, 0, 0, 0.83]
  ],
  "standard_deviations": {
    "time": 0.99,
    "cost": 0.98,
    "walking_distance": 0.91,
    "transfers": 0.91
  },
  "created_at": "2026-08-27T04:54:45Z",
  "updated_at": "2026-08-28T12:39:42Z"
}
```

### 3.5 请求 ID

调用方可以选择在请求头传入：

```http
X-Request-ID: caller-request-001
```

服务会在响应头中原样返回 `X-Request-ID`。如果调用方不传，服务会自动生成 UUID。排查接口错误时可以记录该值。

### 3.6 未声明字段

所有 JSON 请求模型都禁止额外字段。如果传入接口没有定义的属性，会返回 HTTP `422`。

## 4. 健康检查接口

### 4.1 检查 API 进程：`GET /health/live`

用途：只检查 API 进程是否能响应，不检查数据库。

请求参数：无。

```bash
curl "$BASE_URL/health/live"
```

成功状态：HTTP `200`

```json
{
  "status": "ok"
}
```

`status=ok` 表示 API 进程正在运行，但不代表数据库一定可用。

### 4.2 检查完整服务：`GET /health/ready`

用途：检查 API 是否能连接 PostgreSQL，并检查数据库迁移版本是否为当前版本。外部服务在开始联调前应优先调用这个接口。

请求参数：无。

```bash
curl "$BASE_URL/health/ready"
```

准备完成时返回 HTTP `200`：

```json
{
  "status": "ready"
}
```

数据库不可用时返回 HTTP `503`：

```json
{
  "status": "not_ready",
  "reason": "database_unavailable"
}
```

数据库版本没有升级到当前版本时返回 HTTP `503`：

```json
{
  "status": "not_ready",
  "reason": "migration_not_current"
}
```

## 5. 用户与画像接口

### 5.1 创建用户和初始画像：`POST /v1/users`

用途：在 `users` 表创建用户，同时在 `user_profiles` 表创建该用户的初始画像。

#### 请求属性

| 属性路径 | 类型 | 必填 | 作用 |
|---|---|---:|---|
| `external_user_id` | string | 是 | 调用方系统中的用户编号，长度 1～200，必须唯一 |
| `initial_profile` | object | 是 | 初始画像配置 |
| `initial_profile.mode` | string | 是 | `default` 或 `preset` |
| `initial_profile.preset` | string/null | 视模式而定 | 使用的命名预设；`default` 时必须为空，`preset` 时必须提供 |

#### 初始化模式

`default` 表示没有已知偏好，四项初始权重都是 25%：

```json
{
  "external_user_id": "student-system-user-001",
  "initial_profile": {
    "mode": "default"
  }
}
```

`preset` 表示直接使用一个系统内置画像：

```json
{
  "external_user_id": "student-system-user-002",
  "initial_profile": {
    "mode": "preset",
    "preset": "time_priority"
  }
}
```

可用预设：

| `preset` 值 | 时间 | 费用 | 步行距离 | 换乘 | 含义 |
|---|---:|---:|---:|---:|---|
| `balanced` | 25% | 25% | 25% | 25% | 四项均衡 |
| `time_priority` | 70% | 10% | 10% | 10% | 时间优先 |
| `cost_priority` | 10% | 70% | 10% | 10% | 费用优先 |
| `low_walking` | 10% | 10% | 70% | 10% | 少步行优先 |
| `low_transfers` | 10% | 10% | 10% | 70% | 少换乘优先 |

预设只是初始画像，不会锁定。后续调用路线选择学习接口后，预设画像仍会继续更新。

#### curl 示例

```bash
curl -X POST "$BASE_URL/v1/users" \
  -H "Content-Type: application/json" \
  -d '{
    "external_user_id": "student-system-user-001",
    "initial_profile": {
      "mode": "default"
    }
  }'
```

#### 成功响应

成功状态：HTTP `201`

```json
{
  "id": "a5b4f55d-60d3-48b7-896f-5321e1d157b4",
  "external_user_id": "student-system-user-001",
  "initialization_mode": "default",
  "preset_name": null,
  "created_at": "2026-08-31T08:00:00Z",
  "profile": {
    "user_id": "a5b4f55d-60d3-48b7-896f-5321e1d157b4",
    "profile": {
      "evidence_count": 0,
      "converged": true,
      "coefficients": {
        "time": 0,
        "cost": 0,
        "walking_distance": 0,
        "transfers": 0
      },
      "weights": {
        "time": 0.25,
        "cost": 0.25,
        "walking_distance": 0.25,
        "transfers": 0.25
      },
      "percentages": {
        "time": 25,
        "cost": 25,
        "walking_distance": 25,
        "transfers": 25
      },
      "covariance": [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
      ],
      "standard_deviations": {
        "time": 1,
        "cost": 1,
        "walking_distance": 1,
        "transfers": 1
      },
      "created_at": "2026-08-31T08:00:00Z",
      "updated_at": "2026-08-31T08:00:00Z"
    }
  }
}
```

返回属性：

| 属性 | 作用 |
|---|---|
| `id` | 服务生成的内部用户 UUID，后续接口必须使用它 |
| `external_user_id` | 创建时传入的外部用户编号 |
| `initialization_mode` | 实际使用的初始化模式 |
| `preset_name` | 使用的预设名称；默认模式为 `null` |
| `created_at` | 用户创建时间 |
| `profile.user_id` | 当前画像所属的内部用户 UUID，与顶层 `id` 相同 |
| `profile.profile` | 当前画像数据，字段见“用户画像对象” |

可能错误：

- `409 RESOURCE_CONFLICT`：`external_user_id` 已存在。
- `422 REQUEST_VALIDATION_ERROR`：字段缺失、类型错误或模式与预设组合错误。

### 5.2 查询所有用户：`GET /v1/users`

用途：返回系统中全部用户及其当前画像。

请求参数：无。

```bash
curl "$BASE_URL/v1/users"
```

成功状态：HTTP `200`

没有用户时返回：

```json
[]
```

有用户时返回数组：

```json
[
  {
    "id": "a5b4f55d-60d3-48b7-896f-5321e1d157b4",
    "external_user_id": "student-system-user-001",
    "initialization_mode": "default",
    "preset_name": null,
    "created_at": "2026-08-31T08:00:00Z",
    "profile": {
      "evidence_count": 9,
      "converged": true,
      "coefficients": {
        "time": -0.04,
        "cost": -0.51,
        "walking_distance": -1.02,
        "transfers": -0.14
      },
      "weights": {
        "time": 0.023,
        "cost": 0.297,
        "walking_distance": 0.599,
        "transfers": 0.08
      },
      "percentages": {
        "time": 2.3,
        "cost": 29.7,
        "walking_distance": 59.9,
        "transfers": 8.0
      },
      "covariance": [
        [0.98, 0, 0, 0],
        [0, 0.95, 0, 0],
        [0, 0, 0.82, 0],
        [0, 0, 0, 0.83]
      ],
      "standard_deviations": {
        "time": 0.99,
        "cost": 0.98,
        "walking_distance": 0.91,
        "transfers": 0.91
      },
      "created_at": "2026-08-31T08:00:00Z",
      "updated_at": "2026-08-31T09:00:00Z"
    }
  }
]
```

列表项中的 `profile` 已经是画像数据本身，不需要再访问第二层 `profile`：

```text
users[0].profile.weights
```

当前接口没有分页参数，会一次返回全部用户。用户数量较大时需要后续增加分页能力。

### 5.3 查询指定用户画像：`GET /v1/users/{user_id}/profile`

用途：根据内部用户 UUID 查询该用户当前保存的画像。

#### 路径参数

| 参数 | 类型 | 必填 | 作用 |
|---|---|---:|---|
| `user_id` | UUID | 是 | 创建用户接口返回的内部 `id` |

```bash
curl "$BASE_URL/v1/users/a5b4f55d-60d3-48b7-896f-5321e1d157b4/profile"
```

成功状态：HTTP `200`

```json
{
  "user_id": "a5b4f55d-60d3-48b7-896f-5321e1d157b4",
  "profile": {
    "evidence_count": 9,
    "converged": true,
    "coefficients": {
      "time": -0.04,
      "cost": -0.51,
      "walking_distance": -1.02,
      "transfers": -0.14
    },
    "weights": {
      "time": 0.023,
      "cost": 0.297,
      "walking_distance": 0.599,
      "transfers": 0.08
    },
    "percentages": {
      "time": 2.3,
      "cost": 29.7,
      "walking_distance": 59.9,
      "transfers": 8.0
    },
    "covariance": [
      [0.98, 0, 0, 0],
      [0, 0.95, 0, 0],
      [0, 0, 0.82, 0],
      [0, 0, 0, 0.83]
    ],
    "standard_deviations": {
      "time": 0.99,
      "cost": 0.98,
      "walking_distance": 0.91,
      "transfers": 0.91
    },
    "created_at": "2026-08-31T08:00:00Z",
    "updated_at": "2026-08-31T09:00:00Z"
  }
}
```

可能错误：

- `404 RESOURCE_NOT_FOUND`：用户不存在或用户画像不存在。
- `422 REQUEST_VALIDATION_ERROR`：路径中的 `user_id` 不是合法 UUID。

### 5.4 根据路线选择学习画像：`POST /v1/users/{user_id}/choices`

用途：告诉系统“用户选择了哪条路线、拒绝了哪条路线”，系统读取当前画像，以当前后验为下一次先验进行学习，然后同步更新数据库中的当前画像。

这个接口既可以用于初始化阶段的成对路线提问，也可以用于用户真实确认路线后的持续学习。

#### 路径参数

| 参数 | 类型 | 必填 | 作用 |
|---|---|---:|---|
| `user_id` | UUID | 是 | 需要更新画像的内部用户 UUID |

#### 请求属性

| 属性 | 类型 | 必填 | 作用 |
|---|---|---:|---|
| `chosen_route` | 路线对象 | 是 | 用户最终选择的路线，表示用户更偏好它 |
| `rejected_route` | 路线对象 | 是 | 与选中路线同时比较但没有被选择的路线 |

`chosen_route` 和 `rejected_route` 内部的每个属性都使用“路线对象”中定义的结构。

限制：

- 两条路线的 `route_id` 必须不同。
- 两条路线的费用单位必须与服务一致，当前为 `CNY`。
- 如果两条路线四项属性完全相同，算法无法获得有效证据，不更新画像，返回 `learning_applied=false`。
- 一次请求只表达一条成对偏好：`chosen_route` 优于 `rejected_route`。

#### 请求示例

```bash
curl -X POST \
  "$BASE_URL/v1/users/a5b4f55d-60d3-48b7-896f-5321e1d157b4/choices" \
  -H "Content-Type: application/json" \
  -d '{
    "chosen_route": {
      "route_id": "route-cheap-short-walk",
      "total_time_minutes": 42,
      "total_cost": 8,
      "model_cost_unit": "CNY",
      "walking_distance_meters": 180,
      "transfer_count": 2
    },
    "rejected_route": {
      "route_id": "route-fast-expensive",
      "total_time_minutes": 20,
      "total_cost": 30,
      "model_cost_unit": "CNY",
      "walking_distance_meters": 1100,
      "transfer_count": 0
    }
  }'
```

#### 成功响应

成功状态：HTTP `200`

```json
{
  "learning_applied": true,
  "profile": {
    "user_id": "a5b4f55d-60d3-48b7-896f-5321e1d157b4",
    "profile": {
      "evidence_count": 10,
      "converged": true,
      "coefficients": {
        "time": -0.04,
        "cost": -0.56,
        "walking_distance": -1.12,
        "transfers": -0.13
      },
      "weights": {
        "time": 0.022,
        "cost": 0.303,
        "walking_distance": 0.605,
        "transfers": 0.07
      },
      "percentages": {
        "time": 2.2,
        "cost": 30.3,
        "walking_distance": 60.5,
        "transfers": 7.0
      },
      "covariance": [
        [0.98, 0, 0, 0],
        [0, 0.94, 0, 0],
        [0, 0, 0.79, 0],
        [0, 0, 0, 0.82]
      ],
      "standard_deviations": {
        "time": 0.99,
        "cost": 0.97,
        "walking_distance": 0.89,
        "transfers": 0.91
      },
      "created_at": "2026-08-31T08:00:00Z",
      "updated_at": "2026-08-31T09:10:00Z"
    }
  }
}
```

| 返回属性 | 作用 |
|---|---|
| `learning_applied` | `true` 表示本次选择已形成有效证据并保存；`false` 表示路线属性相同，未更新 |
| `profile.user_id` | 被更新的内部用户 UUID |
| `profile.profile` | 更新并保存后的完整当前画像 |

当 `learning_applied=true` 时，可以检查：

- `evidence_count` 比调用前增加 1。
- `updated_at` 发生变化。
- 至少部分 `coefficients`、`weights` 或 `covariance` 发生变化。
- 再调用查询画像接口可以得到相同的更新结果。

重要限制：当前没有幂等键，也不保存选择历史。重复提交同一个请求会被当作新的选择再次学习。调用方必须避免因为网络重试而无意重复提交。

可能错误：

- `404 RESOURCE_NOT_FOUND`：用户或用户画像不存在。
- `422 DOMAIN_VALIDATION_ERROR`：路线编号相同、费用单位错误或路线属性不合法。
- `503 PROFILE_NUMERICAL_FAILURE`：画像数值推断暂时失败。

### 5.5 删除用户：`DELETE /v1/users/{user_id}`

用途：删除指定用户。数据库会级联删除该用户在 `user_profiles` 表中的当前画像。

#### 路径参数

| 参数 | 类型 | 必填 | 作用 |
|---|---|---:|---|
| `user_id` | UUID | 是 | 需要删除的内部用户 UUID |

```bash
curl -X DELETE \
  "$BASE_URL/v1/users/a5b4f55d-60d3-48b7-896f-5321e1d157b4"
```

成功状态：HTTP `204 No Content`，响应没有 JSON 内容。

删除后，再查询该用户画像会返回 `404`。当前项目不保存历史画像，因此删除后不能通过本服务恢复。

可能错误：

- `404 RESOURCE_NOT_FOUND`：用户不存在。
- `422 REQUEST_VALIDATION_ERROR`：`user_id` 不是合法 UUID。

## 6. 路线推荐接口

### 6.1 推荐候选路线：`POST /v1/recommendations`

用途：读取指定用户当前画像，先按本次出行硬限制过滤路线，再对调用方传入的候选路线进行个性化排序并返回 Top-K。

推荐结果是即时计算结果，不会保存路线、推荐结果或推荐会话。

#### 顶层请求属性

| 属性 | 类型 | 必填 | 作用 |
|---|---|---:|---|
| `user_id` | UUID | 是 | 使用哪个用户的当前画像进行推荐 |
| `ranking_mode` | string | 是 | `weighted` 或 `jnd` |
| `routes` | 路线对象数组 | 是 | 待排序的候选路线，至少 1 条，`route_id` 不能重复 |
| `constraints` | object | 否 | 本次出行的硬限制；省略或 `{}` 表示不限制 |
| `top_k` | integer/null | 视模式而定 | `weighted` 模式最终返回数量；必须大于 0；省略表示返回全部可行路线 |
| `jnd` | object/null | 视模式而定 | `jnd` 模式的精排配置 |

#### 模式与参数组合

| `ranking_mode` | `top_k` 写在哪里 | `jnd` 是否允许 | 处理方式 |
|---|---|---|---|
| `weighted` | 顶层 `top_k`，可省略 | 不能传 | 使用画像权重计算加权代价并排序 |
| `jnd` | 必须写在 `jnd.top_k` | 必须传 | 加权初排后，对候选短名单执行 JND 精排 |

错误组合会返回 HTTP `422`。例如：

- `weighted` 模式携带 `jnd`。
- `jnd` 模式没有携带 `jnd`。
- `jnd` 模式仍然在顶层传 `top_k`。

### 6.2 `constraints` 硬限制

`constraints` 在画像排序之前执行。超过限制的路线直接进入 `rejected_routes`，不会参与 Top-K 排序。

| 属性 | 类型 | 必填 | 取值 | 作用 |
|---|---|---:|---|---|
| `max_total_time_minutes` | number/null | 否 | 大于等于 0 | 最长可接受总时间，单位为分钟 |
| `max_total_cost` | number/null | 否 | 大于等于 0 | 最高可接受费用，单位与 `model_cost_unit` 一致 |
| `max_walking_distance_meters` | number/null | 否 | 大于等于 0 | 最长可接受步行距离，单位为米 |
| `max_transfer_count` | integer/null | 否 | 大于等于 0 的整数 | 最多可接受换乘次数 |

示例：

```json
{
  "max_total_time_minutes": 45,
  "max_total_cost": 20,
  "max_walking_distance_meters": 500,
  "max_transfer_count": 1
}
```

限制值本身允许通过。例如 `max_transfer_count=1` 时，换乘 0 次和 1 次都允许，2 次才会被过滤。

`constraints` 只影响本次推荐，不会保存到数据库，也不会更新用户画像。

### 6.3 `weighted` 加权排序

完整请求示例：

```bash
curl -X POST "$BASE_URL/v1/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "a5b4f55d-60d3-48b7-896f-5321e1d157b4",
    "ranking_mode": "weighted",
    "routes": [
      {
        "route_id": "route-fast-direct",
        "total_time_minutes": 18,
        "total_cost": 32,
        "model_cost_unit": "CNY",
        "walking_distance_meters": 1100,
        "transfer_count": 0
      },
      {
        "route_id": "route-economical-short-walk",
        "total_time_minutes": 48,
        "total_cost": 6,
        "model_cost_unit": "CNY",
        "walking_distance_meters": 120,
        "transfer_count": 2
      },
      {
        "route_id": "route-balanced",
        "total_time_minutes": 32,
        "total_cost": 15,
        "model_cost_unit": "CNY",
        "walking_distance_meters": 350,
        "transfer_count": 1
      }
    ],
    "constraints": {},
    "top_k": 2
  }'
```

处理过程：

```text
读取用户当前画像
→ constraints 过滤
→ 路线四维属性固定尺度归一化
→ 归一化属性 × 用户画像权重
→ 四项相加得到 personalized_cost
→ personalized_cost 从小到大排序
→ 返回前 top_k 条
```

当前固定归一化尺度：

| 维度 | 计算方式 |
|---|---|
| 时间 | `total_time_minutes / 180` |
| 费用 | `total_cost / 100` |
| 步行距离 | `walking_distance_meters / 3000` |
| 换乘 | `transfer_count / 4` |

归一化结果不会在 1 处截断，因此超过固定尺度的原始值可能得到大于 1 的结果。

### 6.4 `jnd` 精排

JND 模式适合在加权得分接近时，根据用户最重视的维度和“是否能明显感受到差异”进一步调整排序。

#### `jnd` 属性

| 属性路径 | 类型 | 必填 | 取值限制 | 作用 |
|---|---|---:|---|---|
| `jnd.shortlist_size` | integer | 是 | 大于 0 | 加权初排后，取前多少条进入 JND 精排 |
| `jnd.top_k` | integer | 是 | 大于 0，且不能大于 `shortlist_size` | JND 精排后最终返回多少条 |
| `jnd.thresholds` | object | 是 | 四个比例都必须提供 | 四个维度的 JND 比例阈值 |
| `jnd.thresholds.time_ratio` | number | 是 | 大于等于 0 | 相对短名单最短时间的可感知差异比例 |
| `jnd.thresholds.cost_ratio` | number | 是 | 大于等于 0 | 相对短名单最低费用的可感知差异比例 |
| `jnd.thresholds.walking_distance_ratio` | number | 是 | 大于等于 0 | 相对短名单最短步行距离的可感知差异比例 |
| `jnd.thresholds.transfers_ratio` | number | 是 | 大于等于 0 | 相对短名单最少换乘次数的可感知差异比例 |

例如 `time_ratio=0.1`：如果短名单最短时间是 20 分钟，则时间差异阈值为 2 分钟。路线比 20 分钟多出不超过 2 分钟时，在当前层级视为差异不明显；超过 2 分钟视为明显差异。

阈值为 0 时，任何正差异都被视为明显。如果某个维度的最优参考值为 0，当前实现计算出的差异阈值也是 0。

完整请求示例：

```bash
curl -X POST "$BASE_URL/v1/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "a5b4f55d-60d3-48b7-896f-5321e1d157b4",
    "ranking_mode": "jnd",
    "routes": [
      {
        "route_id": "route-fast-direct",
        "total_time_minutes": 18,
        "total_cost": 32,
        "model_cost_unit": "CNY",
        "walking_distance_meters": 1100,
        "transfer_count": 0
      },
      {
        "route_id": "route-economical-short-walk",
        "total_time_minutes": 48,
        "total_cost": 6,
        "model_cost_unit": "CNY",
        "walking_distance_meters": 120,
        "transfer_count": 2
      },
      {
        "route_id": "route-balanced",
        "total_time_minutes": 32,
        "total_cost": 15,
        "model_cost_unit": "CNY",
        "walking_distance_meters": 350,
        "transfer_count": 1
      }
    ],
    "constraints": {},
    "jnd": {
      "shortlist_size": 3,
      "top_k": 2,
      "thresholds": {
        "time_ratio": 0.1,
        "cost_ratio": 0.1,
        "walking_distance_ratio": 0.1,
        "transfers_ratio": 0.1
      }
    }
  }'
```

JND 模式不能在顶层同时传 `top_k`。

处理过程：

```text
constraints 过滤
→ 用户画像加权初排
→ 取前 shortlist_size 条
→ 按画像权重从高到低确定属性优先级
→ 以短名单每个维度的最优值作为共同参考
→ 根据 JND 阈值逐层区分路线
→ 返回 jnd.top_k 条
```

如果实际可行路线少于 `shortlist_size`，全部可行路线进入精排；如果实际可行路线少于 `top_k`，只返回实际存在的可行路线。

### 6.5 推荐成功响应

成功状态：HTTP `200`

顶层属性：

| 属性 | 类型 | 作用 |
|---|---|---|
| `profile` | object | 本次排序实际使用的用户画像，结构为 `user_id + profile` |
| `ranking_mode` | string | 实际使用的 `weighted` 或 `jnd` |
| `candidate_count` | integer | 请求中传入的候选路线总数 |
| `feasible_count` | integer | 通过 `constraints` 的路线总数 |
| `ranked_routes` | array | 过滤和排序后的路线，最多返回请求指定的 Top-K 数量 |
| `rejected_routes` | array | 因违反 `constraints` 被排除的路线 |
| `explanation` | object | 本次排序的权重与 JND 过程说明 |

#### `ranked_routes` 每项属性

| 属性路径 | 类型 | 作用 |
|---|---|---|
| `rank` | integer | 最终名次，从 1 开始 |
| `route` | object | 该路线的原始属性 |
| `route.route_id` | string | 路线编号 |
| `route.total_time_minutes` | number | 总时间，分钟 |
| `route.total_cost` | number | 总费用 |
| `route.walking_distance_meters` | number | 步行距离，米 |
| `route.transfer_count` | integer | 换乘次数 |
| `personalized_cost` | number | 四项加权贡献之和；在纯加权模式中越小排名越靠前 |
| `normalized_attributes` | object | 四项原始属性除以固定尺度后的结果 |
| `weighted_contributions` | object | 四项归一化属性分别乘以用户画像权重后的结果 |
| `advantage_dimensions` | string[] | 相比所有可行路线的平均水平，该路线具有加权优势的维度 |

`normalized_attributes` 和 `weighted_contributions` 都包含：

```json
{
  "time": 0,
  "cost": 0,
  "walking_distance": 0,
  "transfers": 0
}
```

`weighted_contributions` 四项之和等于 `personalized_cost`。

在 JND 模式中，`personalized_cost` 是加权初排分数，而 `rank` 是 JND 精排后的最终名次。因此可能出现第 1 名的 `personalized_cost` 略大于第 2 名，这不是计算错误。

#### `rejected_routes` 每项属性

| 属性 | 类型 | 作用 |
|---|---|---|
| `route_id` | string | 被硬限制过滤的路线编号 |
| `violated_dimensions` | string[] | 违反的限制维度，可能包含 `time`、`cost`、`walking_distance`、`transfers` 中的一项或多项 |

#### `explanation`：weighted 模式

```json
{
  "normalized_weights": {
    "time": 0.023,
    "cost": 0.297,
    "walking_distance": 0.599,
    "transfers": 0.08
  }
}
```

| 属性 | 作用 |
|---|---|
| `normalized_weights` | 本次推荐实际使用的四维画像权重，四项之和约为 1 |

#### `explanation`：JND 模式

JND 模式除了 `normalized_weights`，还会返回：

| 属性 | 类型 | 作用 |
|---|---|---|
| `attribute_priority` | string[] | 按用户画像权重从高到低排列的四个比较维度 |
| `reference_values` | object | JND 短名单中四个维度各自的最优原始值 |
| `comparison_steps` | array | JND 每一层实际比较的过程记录 |

`comparison_steps` 每项属性：

| 属性 | 类型 | 作用 |
|---|---|---|
| `priority_level` | integer | 当前比较层级，从 1 开始 |
| `dimension` | string | 当前比较的维度 |
| `route_ids` | string[] | 当前层级参与比较的路线编号 |
| `reference_value` | number | 当前维度的共同最优参考值 |
| `threshold_ratio` | number | 请求中传入的当前维度 JND 比例 |
| `noticeable_difference` | number | `reference_value × threshold_ratio` 得到的实际差异阈值 |
| `within_jnd_route_ids` | string[] | 与最优参考值差异没有超过阈值的路线 |
| `outside_jnd_route_ids` | string[] | 与最优参考值差异超过阈值的路线 |

如果所有路线都被 `constraints` 过滤，接口仍然返回 HTTP `200`：

```json
{
  "candidate_count": 3,
  "feasible_count": 0,
  "ranked_routes": [],
  "rejected_routes": [
    {
      "route_id": "route-a",
      "violated_dimensions": ["cost"]
    }
  ]
}
```

实际响应中仍然会包含 `profile`、`ranking_mode` 和 `explanation`。

可能错误：

- `404 RESOURCE_NOT_FOUND`：`user_id` 对应的用户或画像不存在。
- `422 REQUEST_VALIDATION_ERROR`：字段缺失、类型错误、未知字段、Top-K 非正整数或模式参数组合错误。
- `422 DOMAIN_VALIDATION_ERROR`：路线编号重复、费用单位不一致或路线属性不合法。

## 7. 统一错误响应

除健康就绪接口的 `not_ready` 响应外，业务错误统一返回：

```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "用户不存在",
    "details": {},
    "request_id": "680ad6e5-89be-4f49-ae87-7be6df2c20f8"
  }
}
```

| 属性 | 作用 |
|---|---|
| `error.code` | 稳定的程序错误码，调用方可以用于分支处理 |
| `error.message` | 适合开发联调阅读的错误说明 |
| `error.details` | 具体校验错误，可能是对象或数组 |
| `error.request_id` | 本次请求编号，可用于日志排查 |

常见状态和错误码：

| HTTP 状态 | `error.code` | 含义 |
|---:|---|---|
| 404 | `RESOURCE_NOT_FOUND` | 用户或画像不存在 |
| 409 | `RESOURCE_CONFLICT` | `external_user_id` 已存在 |
| 409 | `DATABASE_CONFLICT` | 请求与数据库现有状态冲突 |
| 422 | `REQUEST_VALIDATION_ERROR` | JSON 字段、类型、取值范围或模式组合不符合接口契约 |
| 422 | `DOMAIN_VALIDATION_ERROR` | 路线编号、费用单位或业务规则不符合要求 |
| 503 | `PROFILE_NUMERICAL_FAILURE` | 画像推断暂时失败 |
| 503 | `DATABASE_UNAVAILABLE` | 数据库暂时不可用 |

## 8. 推荐调用顺序

外部系统建议按照下面顺序接入：

```text
1. GET /health/ready
   确认服务和数据库可用

2. POST /v1/users
   创建用户和初始画像，保存返回的内部 id

3. 可选：多次 POST /v1/users/{user_id}/choices
   通过成对路线提问预先学习画像

4. GET /v1/users/{user_id}/profile
   检查当前画像、权重和 evidence_count

5. POST /v1/recommendations
   传入本次候选路线，获得 weighted 或 JND Top-K

6. 用户最终确认路线后，POST /v1/users/{user_id}/choices
   将最终选择路线作为 chosen_route，将一条实际展示但未选择的路线作为 rejected_route

7. 再次 GET /v1/users/{user_id}/profile
   确认画像已经更新并保存

8. 后续再次推荐时继续使用同一个 user_id
```

## 9. 当前能力边界与调用注意事项

1. 服务不生成候选路线，只排序调用方传入的候选路线。
2. 推荐结果不保存；调用方需要时应自行保存业务侧推荐记录。
3. 用户画像只保存当前状态，不保存画像历史。
4. 选择学习接口一次只接收一条选中路线和一条对比路线。
5. 当前没有幂等键；重复提交选择会重复学习。
6. 当前没有撤回选择或从完整历史重新计算画像的接口。
7. `constraints` 是单次出行硬限制，不会写入画像。
8. 命名预设只是初始值，后续选择仍可改变画像。
9. `weighted` 和 `jnd` 是两种推荐模式；测试完整推荐能力时建议两种都验证。
10. 当前费用单位由 `MODEL_COST_UNIT` 配置，Docker 默认是 `CNY`。所有路线必须使用同一单位。
