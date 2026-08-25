"use strict";

const PAPERS = {
  favour: {
    title: "FAVOUR · Personalized Situation-Aware Multimodal Routes（2017）",
    href: "../paper/个性化推荐服务核心论文/02_FAVOUR_Personalized_Situation_Aware_Multimodal_Routes_2017.pdf",
  },
  planner: {
    title: "Preference-Aware Multimodal Journey Planner（2026）",
    href: "../paper/个性化推荐服务核心论文/01_Preference_Aware_Multimodal_Journey_Planner_2026.pdf",
  },
  jnd: {
    title: "Route Guidance Ranking Procedures with Human Perception Consideration（2020）",
    href: "../paper/2026-8-17/1-s2.0-S0968090X20305829-main.pdf",
  },
};

const FLOWS = {
  profile: {
    label: "画像学习",
    entry: "PairwisePreferenceWeightLearner.fit()",
    files: [
      { path: "examples/jnd_enhanced_route_ranking_demo.py", role: "构造选择并调用公开入口", node: "p0" },
      { path: "src/profile/models.py", role: "输入、后验与返回对象", node: "p1" },
      { path: "src/profile/learner.py", role: "画像流程总编排", node: "p2" },
      { path: "src/profile/normalization.py", role: "四维固定尺度转换", node: "p5" },
      { path: "src/profile/inference.py", role: "似然、Laplace 与 MPP", node: "p4" },
      { path: "src/profile/optimization.py", role: "箱约束 trust-region 求解", node: "p8" },
    ],
    nodes: [
      {
        id: "p0",
        file: "examples/jnd_enhanced_route_ranking_demo.py",
        line: "126–152",
        title: "演示脚本组装输入",
        signature: "profile = PairwisePreferenceWeightLearner().fit(tuple(comparisons))",
        summary: "调用方先创建 RouteAttributes，再把每次“选 A、拒绝 B”包装成 PairwisePreference。演示每累计一题就重新调用一次 fit()，因此输出能展示画像怎样逐步变化。",
        input: "6 个 PairwisePreference；每个对象持有 chosen 与 rejected 两条 RouteAttributes。",
        output: "一个 PreferenceLearningResult；当前演示的 evidence_count = 6，converged = True。",
        next: "进入 learner.py 的公开方法 fit()。",
        why: "入口脚本只负责准备领域对象与展示结果，不包含学习算法。这条边界让核心逻辑可以被测试或被其他服务复用。",
        note: "这里的六次选择是固定演示数据，不是用户真实反馈。脚本最后模拟选择推荐第一名，也没有自动写回画像。",
        code: [
          [130, "learner = PairwisePreferenceWeightLearner()"],
          [131, "comparisons: list[PairwisePreference] = []"],
          [134, "comparison = choice.comparison"],
          [135, "comparisons.append(comparison)"],
          [136, "current_profile = learner.fit(tuple(comparisons))", true],
        ],
        paper: {
          ...PAPERS.favour,
          detail: "论文给出“群体先验 → 成对选择 → 个人后验”的流程；演示脚本只是把当前四维缩减实现串起来。",
        },
      },
      {
        id: "p1",
        file: "src/profile/models.py",
        line: "82–158",
        title: "数据对象先阻止无效证据",
        signature: "PairwisePreference(chosen: RouteAttributes, rejected: RouteAttributes)",
        summary: "RouteAttributes 在创建时验证 route_id、非负有限数和非负整数；PairwisePreference 再拒绝同一路线或四项属性完全相同的比较。算法因此不会收到“没有信息量”的证据。",
        input: "两条具有时间、费用、步行距离、换乘次数的 RouteAttributes。",
        output: "合法 PairwisePreference；或立即抛出 ProfileValidationError。",
        next: "fit() 接收一组已经通过领域校验的比较对象。",
        why: "验证放在冻结的数据类中，而不是散落在推断循环里。对象一旦创建成功，后续函数就可以相信基本不变量。",
        note: "这些 dataclass 使用 frozen=True 与 slots=True；不是为了算法，而是为了减少运行期误改和无关属性。",
        code: [
          [126, "def __post_init__(self) -> None:"],
          [127, "    if self.chosen.route_id == self.rejected.route_id:"],
          [128, "        raise ProfileValidationError(\"成对比较中的两条路线必须不同\")", true],
          [129, "    if all("],
          [134, "        raise ProfileValidationError(\"成对比较中的路线属性不能完全相同\")", true],
        ],
        paper: {
          title: "领域约束 · 项目实现",
          href: "../src/profile/models.py",
          detail: "论文需要有效的成对偏好证据；具体的 Python 数据校验方式属于本项目的工程防线。",
        },
      },
      {
        id: "p2",
        file: "src/profile/learner.py",
        line: "176–241",
        title: "fit() 是画像侧的总编排器",
        signature: "fit(comparisons, group_histories=(), *, preference_preset=None) -> PreferenceLearningResult",
        summary: "这个方法本身不做数值优化。它依次决定先验、可选地精炼群体 MPP、提取特征、逐题更新后验、预测选择概率，再把负系数转成易读的正权重。",
        input: "个人比较；可选群体历史；或一个命名 preset。",
        output: "PreferenceLearningResult：posterior、weights、evidence_count、converged、choice_probabilities。",
        next: "先执行先验选择分支；若有 group_histories，再进入群体 MPP 精炼。",
        why: "它是 Facade：把六个内部组件隐藏在一个稳定入口后面。调用方不需要知道优化器、Hessian 或矩阵求逆。",
        note: "未指定 preset 时使用标准 N(0, I)；若有 group_histories，再从该起点精炼群体 MPP。",
        code: [
          [203, "if preference_preset is not None:"],
          [204, "    prior = preset_preference_prior(preference_preset)", true],
          [213, "observations = self._extract(comparisons)"],
          [214, "posterior, converged = self._inference.update_incrementally("],
          [235, "return PreferenceLearningResult(", true],
        ],
        paper: {
          ...PAPERS.favour,
          detail: "对应论文图 1 / 图 2 的主流程；四维缩减和 Python Facade 是项目实现。",
        },
      },
      {
        id: "p3",
        file: "src/profile/learner.py",
        line: "81–173 · 203–211",
        title: "先验从两种来源中选择",
        signature: "preset / standard N(0, I) → GaussianPreferenceModel",
        summary: "有 preset 时把正权重归一化并乘以负强度；否则创建均值 0、协方差 I 的标准 MPP。四项代价系数都限制在 [-20, 0]。",
        input: "命名 preset，或什么都不传。",
        output: "GaussianPreferenceModel，包含 mean、covariance、lower_bounds、upper_bounds。",
        next: "若 group_histories 非空，先验交给 MassPreferencePriorEstimator.refine()；否则直接用于个人更新。",
        why: "代价越大效用应越低，所以系数必须非正。展示给人的 weights 则在最后通过 -mean 变回非负敏感度。",
        note: "preset 不是固定标签：它只是贝叶斯先验。后续足够多的真实选择仍能改写它。",
        code: [
          [203, "if preference_preset is not None:"],
          [204, "    prior = preset_preference_prior(preference_preset)", true],
          [206, "    prior = standard_mass_preference_prior()"],
          [163, "mean={dimension: 0.0 for dimension in PREFERENCE_DIMENSIONS},"],
          [165, "lower_bounds={"],
          [166, "    dimension: _COEFFICIENT_LOWER_BOUND"],
          [169, "upper_bounds={"],
          [170, "    dimension: _COEFFICIENT_UPPER_BOUND"],
        ],
        paper: {
          ...PAPERS.favour,
          detail: "N(0, I) MPP 初值与群体先验流程来自论文；[-20, 0] 边界及命名 preset 是项目配置。",
        },
      },
      {
        id: "p4",
        file: "src/profile/inference.py",
        line: "238–261",
        title: "可选分支：先精炼群体 MPP",
        signature: "MassPreferencePriorEstimator.refine(user_training_sets, initial_mpp) -> GaussianPreferenceModel",
        summary: "只有传入 group_histories 才进入这里。每轮对每个历史用户做一次后验推断，再聚合所有后验；新旧群体 Gaussian 的 KL 散度小于 0.001 时结束。",
        input: "至少一个历史用户，且每个用户至少有一条 FeatureComparison。",
        output: "收敛后的群体 GaussianPreferenceModel；最多迭代 100 次。",
        next: "精炼后的群体模型成为当前个人的 prior。",
        why: "它解决冷启动：先让个人从相似群体附近开始，再用自己的选择逐步偏离。代码不负责判断谁属于哪个群体。",
        note: "没有群体历史时不会运行这段循环。超过最大迭代次数或任一用户后验不收敛会抛 ProfileNumericalError。",
        code: [
          [249, "current = initial_mpp"],
          [250, "for _ in range(self._MAX_ITERATIONS):"],
          [253, "    posterior, converged = self._inference.infer(training_set, current)"],
          [257, "refined = self.aggregate(posteriors)"],
          [258, "if self._kl_divergence(current, refined) < self._KL_TOLERANCE:", true],
          [259, "    return refined"],
        ],
        paper: {
          ...PAPERS.favour,
          detail: "对应公式（7）与图 2 的群体先验聚合与迭代；群体匹配规则未在论文中完整公开，因此留给上游。",
        },
      },
      {
        id: "p5",
        file: "src/profile/normalization.py",
        line: "18–44",
        title: "原始路线变成可比较的四维向量",
        signature: "NormalizedCostFeatureExtractor.extract_comparison(preference) -> FeatureComparison",
        summary: "chosen 与 rejected 的时间、费用、步行、换乘分别除以 180、100、3000、4。它不做裁剪，所以 500 分钟不会和 200 分钟被压成同一个特征。",
        input: "PairwisePreference，内部仍是有单位的 RouteAttributes。",
        output: "FeatureComparison(chosen: Vector, rejected: Vector)。",
        next: "utility_difference() 计算 chosen - rejected，作为 Logit 与后验目标函数的特征差。",
        why: "四个指标量纲不同；固定尺度同时被学习与推荐复用，保证同一权重在不同候选集合中意义一致。",
        note: "固定尺度是业务参数，不是 FAVOUR 论文给出的常数；这是当前项目为跨请求稳定性做的选择。",
        code: [
          [19, "NORMALIZATION_SCALES = {"],
          [21, "    PreferenceDimension.TIME: 180.0,"],
          [22, "    PreferenceDimension.COST: 100.0,"],
          [23, "    PreferenceDimension.WALKING_DISTANCE: 3000.0,"],
          [24, "    PreferenceDimension.TRANSFERS: 4.0,"],
          [36, "route.value_for(dimension) / NORMALIZATION_SCALES[dimension]", true],
        ],
        paper: {
          ...PAPERS.planner,
          detail: "论文要求不同量纲先归一化；当前固定尺度与“不裁剪”策略属于项目选择。",
        },
      },
      {
        id: "p6",
        file: "src/profile/inference.py",
        line: "131–146",
        title: "每一题的后验成为下一题先验",
        signature: "FavourLaplaceInference.update_incrementally(observations, prior) -> (posterior, converged)",
        summary: "循环不是一次性把六题塞进同一个优化调用，而是每次只用一个 observation 调 infer()。该题得到的 posterior 立刻赋给 posterior 变量，下一轮把它当 prior。",
        input: "FeatureComparison 序列 + 初始 GaussianPreferenceModel。",
        output: "最后一题之后的 GaussianPreferenceModel 与累计收敛标记。",
        next: "每轮调用 infer((observation,), posterior)。",
        why: "这直接表达 FAVOUR 的在线更新语义：新证据到来时无需重写外部接口，当前后验就是下一时刻先验。",
        note: "如果某一步不收敛，代码立即停止；它不会把不可靠后验继续传下去。",
        code: [
          [138, "posterior = prior"],
          [140, "for observation in observations:"],
          [141, "    posterior, step_converged = self.infer((observation,), posterior)", true],
          [142, "    if not step_converged:"],
          [144, "        raise ProfileNumericalError(\"增量后验更新未收敛\")"],
          [146, "return posterior, converged"],
        ],
        paper: {
          ...PAPERS.favour,
          detail: "对应 FAVOUR 公式（6）的逐题贝叶斯更新。",
        },
      },
      {
        id: "p7",
        file: "src/profile/inference.py",
        line: "54–92",
        title: "目标函数把先验与选择证据相加",
        signature: "FavourPosteriorObjective.evaluate(coefficients) -> ObjectiveEvaluation",
        summary: "先计算 Gaussian 先验的二次惩罚，再对每条 observation 加入 Bradley–Terry Logit 的负对数似然；同时解析计算 value、gradient 与 Hessian。",
        input: "当前四维系数向量。",
        output: "ObjectiveEvaluation(value, gradient, hessian)，直接供 trust-region 优化器使用。",
        next: "BoxBoundedTrustRegionOptimizer.optimize() 寻找边界内的后验众数。",
        why: "解析梯度与 Hessian 让优化器不依赖数值差分；Hessian 随后还会被求逆，用来形成 Laplace 协方差。",
        note: "这里优化的是负对数后验，所以返回值越小越好；先验精度矩阵来自 covariance 的逆。",
        code: [
          [73, "gradient = list(matrix_vector_product(self._precision, centered))"],
          [75, "value = 0.5 * quadratic_form(centered, self._precision)"],
          [77, "for observation in self._observations:"],
          [79, "    terms = self._likelihood.evaluate(dot(coefficients, difference))", true],
          [88, "return ObjectiveEvaluation("],
          [91, "    hessian=symmetrize(tuple(tuple(row) for row in hessian)),"],
        ],
        paper: {
          ...PAPERS.favour,
          detail: "对应成对 Logit 概率、联合似然与 Gaussian 先验形成的后验目标（公式 4–5）。",
        },
      },
      {
        id: "p8",
        file: "src/profile/optimization.py",
        line: "352–385",
        title: "五个随机起点求边界内最优解",
        signature: "BoxBoundedTrustRegionOptimizer.optimize(objective, lower_bounds, upper_bounds) -> OptimizationResult",
        summary: "优化器先校验上下界，再用固定随机种子生成五个起点。单个起点失败不会终止；只要有收敛结果，就选目标函数值最小的一个。",
        input: "objective.evaluate 回调与四维上下界。",
        output: "最佳收敛 OptimizationResult；全部失败则抛 ProfileNumericalError。",
        next: "infer() 对最佳点的 Hessian 求逆，构造 posterior covariance。",
        why: "多起点降低局部失败风险；固定种子让测试与演示可复现；边界确保四项代价系数保持非正。",
        note: "这是纯 Python 的无外部依赖实现。项目没有依赖 SciPy，数值失败会以领域异常暴露，而不是悄悄返回坏结果。",
        code: [
          [368, "random = Random(self._SEED)"],
          [369, "starts = tuple("],
          [375, "for start in starts:"],
          [377, "    results.append(self._run(objective, start, lower, upper))"],
          [382, "converged = [result for result in results if result.converged]"],
          [384, "    raise ProfileNumericalError(\"所有随机起点均未得到收敛结果\")", true],
          [385, "return min(converged, key=lambda result: result.evaluation.value)"],
        ],
        paper: {
          ...PAPERS.favour,
          detail: "箱约束 trust-region、多起点与解析导数按项目的 FAVOUR 实现说明落地。",
        },
      },
      {
        id: "p9",
        file: "src/profile/inference.py",
        line: "101–129",
        title: "最优点与 Hessian 变成 Laplace 后验",
        signature: "FavourLaplaceInference.infer(observations, prior) -> (GaussianPreferenceModel, bool)",
        summary: "没有 observation 时原样返回 prior。否则优化得到后验众数，用最优点处 Hessian 的逆作为协方差，再沿用先验的系数边界创建新的 GaussianPreferenceModel。",
        input: "当前题的 FeatureComparison 与上一时刻 Gaussian prior。",
        output: "新的 GaussianPreferenceModel(mean, covariance, bounds)。",
        next: "update_incrementally() 把它作为下一题先验；最后交回 fit()。",
        why: "Laplace 近似不只给一个权重点估计，还保留不确定性；后面的选择概率会使用 covariance，而不是只看 mean。",
        note: "covariance 必须严格正定，GaussianPreferenceModel 的构造会再次做 Cholesky 校验。",
        code: [
          [107, "if not observations:"],
          [108, "    return prior, True"],
          [110, "objective = FavourPosteriorObjective(observations, prior)"],
          [111, "optimized = self._optimizer.optimize("],
          [119, "covariance = inverse_matrix(optimized.evaluation.hessian)", true],
          [120, "posterior = GaussianPreferenceModel("],
        ],
        paper: {
          ...PAPERS.favour,
          detail: "对应后验众数与 Laplace Gaussian 协方差近似。",
        },
      },
      {
        id: "p10",
        file: "src/profile/learner.py",
        line: "218–241",
        title: "后验被包装成可消费的画像结果",
        signature: "posterior → choice_probabilities + weights → PreferenceLearningResult",
        summary: "先用 posterior mean 与 covariance 计算每条已选路线的后验选择概率；再取 max(0, -mean) 作为四维敏感度并归一化。若总敏感度接近 0，则展示权重回退为四项各 25%。",
        input: "最终 posterior 与全部 FeatureComparison。",
        output: "公开返回对象 PreferenceLearningResult。",
        next: "推荐模块可直接把整个 result 作为 preference 传入 rank()。",
        why: "论文效用仍使用非正系数 result.utility_coefficients；result.weights 只是把方向翻转并归一化后的易读接口。两者不能混为一谈。",
        note: "选择概率使用 covariance 做不确定性衰减；weights 只使用 mean。因此结果同时服务“解释权重”和“评估证据可信度”两种读取方式。",
        code: [
          [218, "probabilities = tuple("],
          [219, "    self._predictor.probability(posterior, observation)"],
          [222, "sensitivities = {"],
          [223, "    dimension: max(0.0, -posterior.mean[dimension])", true],
          [227, "if total <= 1e-12:"],
          [235, "return PreferenceLearningResult("],
        ],
        paper: {
          ...PAPERS.favour,
          detail: "后验选择概率对应公式（9）；把 -mean 归一化成展示百分比是项目的接口层设计。",
        },
      },
    ],
  },
  recommendation: {
    label: "路线推荐",
    entry: "JndEnhancedRouteRanker.rank()",
    files: [
      { path: "examples/jnd_enhanced_route_ranking_demo.py", role: "组装候选、阈值与约束", node: "r0" },
      { path: "src/recommendation/jnd.py", role: "JND 总编排与递归精排", node: "r1" },
      { path: "src/recommendation/ranking.py", role: "硬约束与加权初排", node: "r2" },
      { path: "src/recommendation/models.py", role: "配置、明细与返回对象", node: "r3" },
      { path: "src/profile/normalization.py", role: "复用画像侧固定尺度", node: "r4" },
    ],
    nodes: [
      {
        id: "r0",
        file: "examples/jnd_enhanced_route_ranking_demo.py",
        line: "278–305",
        title: "调用方显式交齐本次推荐参数",
        signature: "JndEnhancedRouteRanker().rank(candidates, profile, constraints=…, thresholds=…, shortlist_size=3, top_k=2)",
        summary: "演示先用 fit() 得到 profile，再创建 4 条候选、4 个最大可接受值、4 个 JND 比例，最后请求初排 Top-3 进入精排并返回 Top-2。",
        input: "RouteAttributes[] + PreferenceLearningResult + RouteConstraints + JndThresholds + N/K。",
        output: "JndEnhancedRankingResult。",
        next: "进入 jnd.py 的公开 rank()；它先委托 PersonalizedRouteRanker 做完整初排。",
        why: "阈值没有默认值，调用方必须承认这是业务假设。代码不会把未经验证的感知阈值偷偷写死。",
        note: "shortlist_size = 3、top_k = 2 是演示配置，不是论文常数。",
        code: [
          [281, "constraints = RouteConstraints("],
          [287, "thresholds = JndThresholds("],
          [293, "result = JndEnhancedRouteRanker().rank(", true],
          [298, "    shortlist_size=3,"],
          [299, "    top_k=2,"],
        ],
        paper: {
          title: "组合流程 · 项目实现",
          href: "../src/recommendation/README.md",
          detail: "加权初排与修正 JND 各有论文依据；先 Top-N 再 JND 的组合顺序是本项目为复用能力做的设计。",
        },
      },
      {
        id: "r1",
        file: "src/recommendation/jnd.py",
        line: "146–164",
        title: "JND 入口先复用完整加权排序",
        signature: "rank(routes, preference, *, thresholds, shortlist_size, top_k, constraints=None) -> JndEnhancedRankingResult",
        summary: "先验证 N 与 K 都是正整数且 K ≤ N，再调用 weighted_ranker.rank(top_k=None)。这里故意不提前截断，以便 weighted_result 保留全部可行路线和淘汰记录。",
        input: "候选路线、画像或权重、硬约束、JND 阈值、N 与 K。",
        output: "首先得到完整 RouteRankingResult，随后才开始 JND。",
        next: "进入 PersonalizedRouteRanker.rank() 的输入校验与硬约束过滤。",
        why: "JND 依赖稳定初排顺序作为最终兜底，也需要先排除不可接受路线；因此加权阶段不是可省略的预处理。",
        note: "top_k=None 只在内部调用中使用，保证初排不丢数据；最终 K 在 JND 结束后截取。",
        code: [
          [158, "self._validate_sizes(shortlist_size, top_k)"],
          [159, "weighted_result = self._weighted_ranker.rank(", true],
          [160, "    routes,"],
          [161, "    preference,"],
          [163, "    top_k=None,"],
        ],
        paper: {
          ...PAPERS.jnd,
          detail: "硬约束、加权排序与修正 JND 分别来自论文；两阶段封装方式属于项目架构。",
        },
      },
      {
        id: "r2",
        file: "src/recommendation/ranking.py",
        line: "36–93",
        title: "先把输入变成确定、合法的排序问题",
        signature: "PersonalizedRouteRanker.rank(routes, preference, *, constraints=None, top_k=None)",
        summary: "候选先固化为 tuple；空列表、重复 route_id、非法 top_k 会抛错。preference 可以是完整 PreferenceLearningResult 或四维映射，最终都会校验完整性、非负、有限并归一化到和为 1。",
        input: "至少一条 route_id 唯一的路线；完整四维非负权重。",
        output: "确定的 candidates、normalized weights 与 active constraints。",
        next: "逐条调用 RouteConstraints.violations(route)。",
        why: "权重归一化让调用方可以传 0.4 或 40；route_id 唯一确保平分时的 tie-breaker 真正稳定。",
        note: "bool 在 Python 中是 int 的子类，因此代码显式拒绝 top_k=True 这类看似整数的非法输入。",
        code: [
          [85, "candidates = tuple(routes)"],
          [86, "if not candidates:"],
          [87, "    raise RecommendationValidationError(\"候选路线不能为空\")", true],
          [89, "if len(set(route_ids)) != len(route_ids):"],
          [92, "weights = self._resolve_weights(preference)"],
          [93, "active_constraints = constraints or RouteConstraints()"],
        ],
        paper: {
          ...PAPERS.planner,
          detail: "四维权重进入 WSM；具体输入类型兼容与异常设计属于项目实现。",
        },
      },
      {
        id: "r3",
        file: "src/recommendation/models.py",
        line: "30–97",
        title: "硬约束发生在所有软排序之前",
        signature: "RouteConstraints.violations(route) -> tuple[PreferenceDimension, …]",
        summary: "四个字段都是最大可接受值。每条路线逐项比较，所有超限维度都会被收集；只要非空，就创建 RejectedRoute 并 continue，不计算归一化代价。",
        input: "RouteAttributes 与可选的四项上限。",
        output: "空 tuple 表示可行；非空 tuple 精确记录违反的维度。",
        next: "只有通过约束的路线进入 _normalized_attributes() 与加权计分。",
        why: "硬约束表达“不能接受”，画像权重表达“更偏好”。把两者混成一个惩罚分会让极低加权代价掩盖不可接受条件。",
        note: "被淘汰路线仍保存在结果中，前端或日志可以解释为什么没推荐，而不是只看到它消失。",
        code: [
          [64, "def violations(self, route: RouteAttributes) -> tuple[PreferenceDimension, ...]:"],
          [67, "    violated: list[PreferenceDimension] = []"],
          [76, "    if route.walking_distance_meters > self.max_walking_distance_meters:"],
          [79, "        violated.append(PreferenceDimension.WALKING_DISTANCE)", true],
          [85, "    return tuple(violated)"],
        ],
        paper: {
          ...PAPERS.jnd,
          detail: "对应个人最大/最小可接受值过滤（项目四项均为代价，所以只实现最大值）。",
        },
      },
      {
        id: "r4",
        file: "src/recommendation/ranking.py",
        line: "104–117",
        title: "可行路线按同一尺度计算加权代价",
        signature: "cost(route) = Σ normalized_attribute[d] × weight[d]",
        summary: "每项原始属性除以画像侧同一 NORMALIZATION_SCALES，再乘对应权重；四项 contribution 求和得到 personalized_cost。代价越小越好。",
        input: "一条可行 RouteAttributes 与和为 1 的四维权重。",
        output: "原始路线、总代价、归一化属性、四项贡献的内部 tuple。",
        next: "按 (personalized_cost, route_id) 升序排序。",
        why: "结果不仅保存总分，还保存四项贡献；这样推荐理由可以回到具体维度，而不是一个无法解释的黑箱分数。",
        note: "推荐阶段没有用候选集 min-max，因为候选集变化会改变已学权重的含义；它刻意复用训练尺度。",
        code: [
          [104, "for route in candidates:"],
          [105, "    violations = active_constraints.violations(route)"],
          [110, "    normalized = self._normalized_attributes(route)"],
          [111, "    contributions = {"],
          [112, "        dimension: weights[dimension] * normalized[dimension]", true],
          [115, "    scored.append((route, sum(contributions.values()), normalized, contributions))"],
        ],
        paper: {
          ...PAPERS.planner,
          detail: "对应公式（1）至（4）的加权和模型；固定尺度复用是项目为语义稳定性做的调整。",
        },
      },
      {
        id: "r5",
        file: "src/recommendation/ranking.py",
        line: "117–163",
        title: "初排既给名次，也保留解释材料",
        signature: "scored.sort(key=(cost, route_id)) → RouteRankingResult",
        summary: "先按总代价、再按 route_id 排序；对全部可行路线计算各维平均归一化代价。每条 RankedRoute 保存 advantage_dimensions，表示它在哪些高权重维度优于可行集平均值。",
        input: "所有可行路线的计分 tuple。",
        output: "RouteRankingResult：ranked_routes、rejected_routes、normalized_weights、candidate_count、feasible_count。",
        next: "JND 入口从完整初排中截取前 shortlist_size 条。",
        why: "route_id 是确定性平分规则；advantage_dimensions 用全体可行路线平均值计算，而不是只对最终 Top-K 计算，避免解释随截断漂移。",
        note: "若 top_k 单独用于 PersonalizedRouteRanker，只截 ranked_routes，不改变 feasible_count。JND 内部调用则始终 top_k=None。",
        code: [
          [117, "scored.sort(key=lambda item: (item[1], item[0].route_id))", true],
          [118, "feasible_count = len(scored)"],
          [120, "averages = {"],
          [132, "advantages = sorted("],
          [147, "RankedRoute("],
          [157, "return RouteRankingResult("],
        ],
        paper: {
          ...PAPERS.planner,
          detail: "WSM 决定总代价；稳定 tie-breaker、优势维度与可审计返回结构是项目实现。",
        },
      },
      {
        id: "r6",
        file: "src/recommendation/jnd.py",
        line: "43–68 · 165–184",
        title: "权重只决定 JND 的指标优先级",
        signature: "priority = dimensions sorted by -weight; reference[d] = min(shortlist[d])",
        summary: "四维按权重降序排列；权重相同时保持 TIME、COST、WALKING_DISTANCE、TRANSFERS 的定义顺序。随后只取初排前 N 条，并为每个维度找共同最优原始值。",
        input: "normalized_weights 与初排前 N 条 RankedRoute。",
        output: "attribute_priority、shortlist、reference_values。",
        next: "从最高优先级开始调用 _rerank_group(shortlist, priority_index=0)。",
        why: "权重在 JND 阶段不再乘路线属性；它只回答“先看哪个指标”。真正的可感知差异使用原始分钟、元、米、次数。",
        note: "只有 Top-N 参与 reference_values；被硬约束过滤的路线和初排 N 名以后的路线都不会改变 JND 参照。",
        code: [
          [51, "return tuple("],
          [52, "    sorted("],
          [54, "        key=lambda dimension: ("],
          [55, "    -weights[dimension],"],
          [66, "dimension: min(item.route.value_for(dimension) for item in shortlist)", true],
          [165, "priority = self._attribute_priority(weighted_result.normalized_weights)"],
          [166, "shortlist = weighted_result.ranked_routes[:shortlist_size]"],
        ],
        paper: {
          ...PAPERS.jnd,
          detail: "共同最优参照避免简单两两比较造成不传递；权重排序决定字典序优先级。",
        },
      },
      {
        id: "r7",
        file: "src/recommendation/jnd.py",
        line: "71–144",
        title: "递归把“感觉不出差别”的路线继续往下比",
        signature: "_rerank_group(routes, priority, priority_index, reference_values, thresholds, …) -> list[RankedRoute]",
        summary: "当前维度的允许差值 = 阈值比例 × 共同最优值。difference > limit 的路线进入 noticeable；其余进入 indistinguishable。先递归处理 indistinguishable，再把 noticeable 按当前原始属性升序接在后面。",
        input: "当前比较组、指标优先级位置、共同参照、阈值，以及审计记录容器。",
        output: "当前组的重排列表；同时追加 JndComparisonStep 和首次 decisive_dimension。",
        next: "递归到组内只剩一条，或四个维度都用完。",
        why: "“不可区分组优先”保证仍在 JND 最优范围内的路线继续争夺；完全相同时保留输入的加权顺序作为稳定后备。",
        note: "边界使用严格大于号：差值等于 limit 仍视为不可明显区分。noticeable 中属性值相同的子组才会进入下一指标。",
        code: [
          [86, "dimension = priority[priority_index]"],
          [88, "noticeable_limit = thresholds.ratio_for(dimension) * reference"],
          [93, "difference = item.route.value_for(dimension) - reference"],
          [94, "if difference > noticeable_limit:", true],
          [116, "ordered = cls._rerank_group("],
          [128, "noticeable.sort(key=lambda item: item.route.value_for(dimension))"],
        ],
        paper: {
          ...PAPERS.jnd,
          detail: "对应修正 JND 比较与 Algorithm 1；comparison_steps 是项目为解释性增加的审计记录。",
        },
      },
      {
        id: "r8",
        file: "src/recommendation/jnd.py",
        line: "168–207",
        title: "最后才截 Top-K，并把全过程一起返回",
        signature: "reranked[:top_k] → JndEnhancedRankingResult",
        summary: "无 shortlist 时返回合法空结果。正常情况下用 dataclasses.replace 只更新最终路线的 rank，再把 weighted_result、priority、thresholds、reference_values、decisive_dimensions 与 comparison_steps 一并封装。",
        input: "JND 重排后的 shortlist 与前面累积的审计信息。",
        output: "JndEnhancedRankingResult；recommended 属性返回第一条，没有则为 None。",
        next: "调用方展示 Top-K，也可以检查初排、淘汰原因和每轮 JND 过程。",
        why: "最终结果不只是一串 route_id。它保留“为什么过滤、怎么计分、在哪个指标被区分”的完整证据链。",
        note: "演示结果中 weighted_result 第一名是 slow-cheap，而最终 recommended 是 balanced；两种顺序都保留，因此变化可核查。",
        code: [
          [168, "if not shortlist:"],
          [169, "    return JndEnhancedRankingResult("],
          [194, "final_routes = tuple("],
          [195, "    replace(item, rank=rank)"],
          [196, "    for rank, item in enumerate(reranked[:top_k], start=1)", true],
          [198, "return JndEnhancedRankingResult("],
          [200, "    weighted_result=weighted_result,"],
        ],
        paper: {
          title: "可解释返回结构 · 项目实现",
          href: "../src/recommendation/models.py",
          detail: "论文定义排序逻辑；完整保留中间结果、决定维度和比较步骤是本项目为验证与前端解释增加的结构。",
        },
      },
    ],
  },
};

const SECTION_ENTRIES = [
  { title: "对象生命周期", subtitle: "RouteAttributes 到 JndEnhancedRankingResult", hash: "#objects", aliases: "数据 对象 dataclass 生命周期" },
  { title: "真实运行一次", subtitle: "六次选择、过滤、初排与 JND 反转", hash: "#runtime", aliases: "demo 演示 balanced slow-cheap" },
  { title: "分支与异常", subtitle: "空数据、无可行路线和不收敛", hash: "#branches", aliases: "error exception validation 边界" },
  { title: "论文到函数映射", subtitle: "FAVOUR、WSM 与修正 JND", hash: "#evidence", aliases: "paper 参考文献 依据" },
  { title: "当前实现边界", subtitle: "明确哪些能力尚未实现", hash: "#boundaries", aliases: "没有做 范围 上游" },
];

const state = {
  flow: "profile",
  node: "p0",
  searchResults: [],
  searchIndex: 0,
};

const fileList = document.querySelector("#file-list");
const traceList = document.querySelector("#trace-list");
const traceCount = document.querySelector("#trace-count");
const functionDetail = document.querySelector("#function-detail");
const fileIndexSummary = document.querySelector("#file-index-summary");
const atlas = document.querySelector("#atlas");
const appShell = document.querySelector("#app-shell");
const commandDialog = document.querySelector("#command-dialog");
const commandInput = document.querySelector("#command-input");
const commandResults = document.querySelector("#command-results");
const commandStatus = document.querySelector("#command-status");
const searchTrigger = document.querySelector("#search-trigger");

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}

function sourceHref(path) {
  return `../${encodeURI(path)}`;
}

function currentFlow() {
  return FLOWS[state.flow];
}

function currentNode() {
  return currentFlow().nodes.find((node) => node.id === state.node) ?? currentFlow().nodes[0];
}

function renderFiles() {
  const flow = currentFlow();
  fileIndexSummary.textContent = `${flow.label}调用链涉及 ${flow.files.length} 个文件；数字表示该文件在主调用链中的步骤数。`;
  fileList.innerHTML = flow.files
    .map((file) => {
      const count = flow.nodes.filter((node) => node.file === file.path).length;
      const isActive = currentNode().file === file.path;
      return `
        <button class="file-button${isActive ? " is-active" : ""}" type="button" data-node="${file.node}" ${isActive ? 'aria-current="true"' : ""}>
          <span class="file-button__copy">
            <code>${escapeHtml(file.path)}</code>
            <small>${escapeHtml(file.role)}</small>
          </span>
          <span class="file-button__count">${count}</span>
        </button>
      `;
    })
    .join("");
}

function renderTrace() {
  const flow = currentFlow();
  traceCount.textContent = `${flow.nodes.length} 步`;
  traceList.innerHTML = flow.nodes
    .map((node, index) => {
      const isActive = node.id === state.node;
      return `
        <li class="trace-step">
          <button class="trace-step__button${isActive ? " is-active" : ""}" type="button" data-node="${node.id}" aria-pressed="${isActive}">
            <span class="trace-step__number">${String(index + 1).padStart(2, "0")}</span>
            <span class="trace-step__copy">
              <b>${escapeHtml(node.title)}</b>
              <code>${escapeHtml(node.file.split("/").at(-1))}:${escapeHtml(node.line)}</code>
            </span>
          </button>
        </li>
      `;
    })
    .join("");
}

function renderSource(node) {
  const lines = node.code
    .map(([number, code, signal]) => {
      const content = `<span class="line-number">${String(number).padStart(3, " ")}</span>  ${escapeHtml(code)}`;
      return signal ? `<span class="code-signal">${content}</span>` : content;
    })
    .join("\n");

  return `
    <figure class="source-extract">
      <figcaption>关键源码行 · ${escapeHtml(node.file)}:${escapeHtml(node.line)}</figcaption>
      <pre><code>${lines}</code></pre>
    </figure>
  `;
}

function renderDetail() {
  const node = currentNode();
  functionDetail.innerHTML = `
    <header class="detail-head">
      <a class="detail-location" href="${sourceHref(node.file)}">${escapeHtml(node.file)}:${escapeHtml(node.line)} ↗</a>
      <h3>${escapeHtml(node.title)}</h3>
      <code class="detail-signature">${escapeHtml(node.signature)}</code>
      <p class="detail-summary">${escapeHtml(node.summary)}</p>
    </header>

    <dl class="detail-contract">
      <div>
        <dt>输入</dt>
        <dd>${escapeHtml(node.input)}</dd>
      </div>
      <div>
        <dt>输出</dt>
        <dd>${escapeHtml(node.output)}</dd>
      </div>
      <div>
        <dt>下一跳</dt>
        <dd>${escapeHtml(node.next)}</dd>
      </div>
      <div>
        <dt>为什么这样写</dt>
        <dd>${escapeHtml(node.why)}</dd>
      </div>
    </dl>

    ${renderSource(node)}

    <div class="detail-note">
      <span class="detail-note__mark" aria-hidden="true">!</span>
      <p><strong>读代码时别漏掉：</strong>${escapeHtml(node.note)}</p>
    </div>

    <div class="detail-paper">
      <strong>这一步的依据</strong>
      <p>${escapeHtml(node.paper.detail)}</p>
      <a href="${node.paper.href}">${escapeHtml(node.paper.title)} ↗</a>
    </div>
  `;
}

function updateFlowControls() {
  document.querySelectorAll("[data-flow-select]").forEach((button) => {
    const isActive = button.dataset.flowSelect === state.flow;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
  });
}

function renderAtlas() {
  updateFlowControls();
  renderFiles();
  renderTrace();
  renderDetail();
}

function selectNode(nodeId, options = {}) {
  const node = currentFlow().nodes.find((item) => item.id === nodeId);
  if (!node) return;
  state.node = node.id;
  renderAtlas();
  if (options.focusDetail) {
    functionDetail.setAttribute("tabindex", "-1");
    functionDetail.focus({ preventScroll: true });
  }
}

function selectFlow(flowId, options = {}) {
  if (!FLOWS[flowId]) return;
  state.flow = flowId;
  state.node = options.node ?? FLOWS[flowId].nodes[0].id;
  renderAtlas();
  if (options.scroll) {
    atlas.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

document.addEventListener("click", (event) => {
  const flowButton = event.target.closest("[data-flow-select]");
  if (flowButton) {
    selectFlow(flowButton.dataset.flowSelect, {
      scroll: flowButton.classList.contains("entry-row"),
    });
    return;
  }

  const nodeButton = event.target.closest("[data-node]");
  if (nodeButton && !nodeButton.classList.contains("command-result")) {
    selectNode(nodeButton.dataset.node);
  }
});

function allSearchEntries() {
  const nodes = Object.entries(FLOWS).flatMap(([flowId, flow]) =>
    flow.nodes.map((node) => ({
      type: "node",
      flow: flowId,
      node: node.id,
      title: node.title,
      subtitle: `${node.file}:${node.line}`,
      aliases: `${flow.label} ${flow.entry} ${node.signature} ${node.summary}`,
    })),
  );
  const sections = SECTION_ENTRIES.map((entry) => ({ type: "section", ...entry }));
  return [...nodes, ...sections];
}

function normalized(value) {
  return String(value).toLocaleLowerCase("zh-CN").replaceAll(/\s+/g, " ").trim();
}

function filterSearchEntries(query) {
  const needle = normalized(query);
  const entries = allSearchEntries();
  if (!needle) return entries.slice(0, 12);
  return entries
    .filter((entry) => normalized(`${entry.title} ${entry.subtitle} ${entry.aliases}`).includes(needle))
    .slice(0, 18);
}

function renderSearchResults() {
  state.searchResults = filterSearchEntries(commandInput.value);
  if (state.searchIndex >= state.searchResults.length) state.searchIndex = 0;
  commandStatus.textContent = state.searchResults.length
    ? `找到 ${state.searchResults.length} 项结果`
    : "没有匹配结果";

  if (!state.searchResults.length) {
    commandResults.innerHTML = `
      <div class="command-empty">
        <strong>没有匹配项</strong>
        <span>试试函数名、文件名，或“异常”“论文”“对象”。</span>
      </div>
    `;
    return;
  }

  commandResults.innerHTML = state.searchResults
    .map(
      (entry, index) => `
        <button
          class="command-result${index === state.searchIndex ? " is-active" : ""}"
          type="button"
          role="option"
          aria-selected="${index === state.searchIndex}"
          data-search-index="${index}"
        >
          <span class="command-result__copy">
            <b>${escapeHtml(entry.title)}</b>
            <small>${escapeHtml(entry.subtitle)}</small>
          </span>
          <span>${entry.type === "node" ? "函数" : "段落"}</span>
        </button>
      `,
    )
    .join("");
}

function openCommandDialog() {
  if (commandDialog.open) return;
  state.searchIndex = 0;
  commandInput.value = "";
  renderSearchResults();
  commandDialog.showModal();
  appShell.inert = true;
  commandInput.focus();
}

function closeCommandDialog() {
  if (!commandDialog.open) return;
  commandDialog.close();
}

function activateSearchEntry(index) {
  const entry = state.searchResults[index];
  if (!entry) return;
  closeCommandDialog();
  requestAnimationFrame(() => {
    if (entry.type === "node") {
      selectFlow(entry.flow, { node: entry.node });
      atlas.scrollIntoView({ behavior: "smooth", block: "start" });
    } else {
      document.querySelector(entry.hash)?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  });
}

searchTrigger.addEventListener("click", openCommandDialog);

document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLocaleLowerCase("zh-CN") === "k") {
    event.preventDefault();
    commandDialog.open ? closeCommandDialog() : openCommandDialog();
  }
});

commandDialog.addEventListener("close", () => {
  appShell.inert = false;
  searchTrigger.focus({ preventScroll: true });
});

commandDialog.addEventListener("click", (event) => {
  if (event.target === commandDialog) closeCommandDialog();
  const result = event.target.closest("[data-search-index]");
  if (result) activateSearchEntry(Number(result.dataset.searchIndex));
});

commandInput.addEventListener("input", () => {
  state.searchIndex = 0;
  renderSearchResults();
});

commandInput.addEventListener("keydown", (event) => {
  if (!state.searchResults.length) return;
  if (event.key === "ArrowDown") {
    event.preventDefault();
    state.searchIndex = (state.searchIndex + 1) % state.searchResults.length;
    renderSearchResults();
  }
  if (event.key === "ArrowUp") {
    event.preventDefault();
    state.searchIndex = (state.searchIndex - 1 + state.searchResults.length) % state.searchResults.length;
    renderSearchResults();
  }
  if (event.key === "Enter") {
    event.preventDefault();
    activateSearchEntry(state.searchIndex);
  }
});

renderAtlas();
