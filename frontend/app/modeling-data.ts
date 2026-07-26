export const stages = [
  {
    id: "S0",
    label: "定义问题",
    detail: "边界、决策与损失",
    owner: "人 + Agent",
    status: "active",
  },
  {
    id: "S1",
    label: "候选形式化",
    detail: "骨架、假设与淘汰条件",
    owner: "Agent",
    status: "locked",
  },
  {
    id: "S2",
    label: "数据与实验",
    detail: "来源、清洗与切分",
    owner: "Harness",
    status: "locked",
  },
  {
    id: "S3",
    label: "实现与复现",
    detail: "代码、toy test 与重放",
    owner: "Agent + Harness",
    status: "locked",
  },
  {
    id: "S4",
    label: "证伪与恢复",
    detail: "L1–L4、UQ 与换方向",
    owner: "Verifier",
    status: "locked",
  },
  {
    id: "S5",
    label: "预测与决策",
    detail: "冻结输出与适用边界",
    owner: "Harness + 人",
    status: "locked",
  },
  {
    id: "S6",
    label: "报告交付",
    detail: "结果注入与可复现包",
    owner: "Harness",
    status: "locked",
  },
] as const;

export const firstPrincipleQuestions = [
  {
    id: "boundary",
    index: "01",
    title: "系统边界是什么？",
    description: "谁、什么对象、什么时间尺度，以及哪些机制不在本次模型中。",
  },
  {
    id: "memory",
    index: "02",
    title: "状态与记忆在哪里？",
    description: "哪些变量决定未来，历史如何进入状态，是否存在滞后或路径依赖。",
  },
  {
    id: "data",
    index: "03",
    title: "我们能观察到什么？",
    description: "数据来源、采样过程、缺失、噪声、偏差与可追加的实验。",
  },
  {
    id: "loss",
    index: "04",
    title: "什么结果才算有用？",
    description: "最终决策、错误代价、基线、成功阈值与必须拒答的条件。",
  },
] as const;

export const modelingLoop = [
  {
    id: "understand",
    label: "理解现实",
    detail: "把自然语言问题收敛为可计算任务",
    status: "active",
  },
  {
    id: "compete",
    label: "候选竞争",
    detail: "经典骨架、结构变体与生成候选共存",
    status: "queued",
  },
  {
    id: "falsify",
    label: "主动证伪",
    detail: "不是问能否拟合，而是寻找失败证据",
    status: "queued",
  },
  {
    id: "recover",
    label: "图中恢复",
    detail: "根据失败签名补丁、换表征或换方向",
    status: "queued",
  },
  {
    id: "deliver",
    label: "可审计交付",
    detail: "冻结预测、限制声明、报告与复现包",
    status: "queued",
  },
] as const;

export const authorityRoles = [
  {
    role: "Agent",
    tone: "blue",
    action: "提出与执行",
    description: "澄清问题、生成候选、写代码、运行允许的计算、解释证据。",
  },
  {
    role: "Harness",
    tone: "green",
    action: "验证与记账",
    description: "冻结输入、执行类型化检查、控制 Graph 转移、记录失败与恢复。",
  },
  {
    role: "你",
    tone: "amber",
    action: "定义价值与批准",
    description: "决定问题是否值得做、错误代价、数据授权及最终现实行动。",
  },
] as const;

export const completedTask = {
  id: "i36-wb-f078f2e7f23214c3ed07",
  shortId: "I36 · f078…",
  title: "盲化正值序列四步预测",
  status: "PUBLIC ELIGIBLE",
  selectedModel: "log_random_walk_drift",
  predictions: [264.918064, 274.036039, 283.467838, 293.224261],
};

export const graphNodes = [
  {
    id: "custody",
    label: "来源托管",
    status: "VERIFIED",
    tone: "pass",
    title: "双重未见来源已经代码验证",
    description:
      "来源选择发生在协议提交之后；I34 与 I35 在身份、响应字节和来源记录中都被排除。",
    facts: [
      { label: "先前任务排除", value: "I34 + I35" },
      { label: "来源身份", value: "WITHHELD" },
    ],
  },
  {
    id: "primary",
    label: "自主 ODE",
    status: "L3 FAIL",
    tone: "fail",
    title: "初始指数趋势模型没有过科学门",
    description:
      "验证误差、创新相关与区间覆盖没有同时满足冻结阈值，系统保留失败并触发恢复。",
    facts: [
      { label: "候选", value: "exponential.trend_only" },
      { label: "验证相对 RMSE", value: "18.43%" },
    ],
  },
  {
    id: "router",
    label: "换表征",
    status: "TRIGGERED",
    tone: "warn",
    title: "失败触发换方向，而不是放宽阈值",
    description:
      "Graph 根据 primary_l3_fail 生成两个对数增长候选；原任务与验收线保持不变。",
    facts: [
      { label: "恢复候选", value: "2" },
      { label: "阈值变化", value: "0" },
    ],
  },
  {
    id: "ar1",
    label: "增长 AR(1)",
    status: "REJECTED",
    tone: "fail",
    title: "误差最低，但参数稳定性不足",
    description:
      "φ 窗口范围 0.3046 超过冻结上限 0.3000，因此没有因为表现好而越过机制门。",
    facts: [
      { label: "验证相对 RMSE", value: "1.69%" },
      { label: "φ 窗口范围", value: "0.3046" },
    ],
  },
  {
    id: "recovery",
    label: "对数漂移",
    status: "SELECTED",
    tone: "pass",
    title: "唯一通过全部冻结约束的恢复候选",
    description:
      "它不是误差最低者，却同时满足误差、提升、稳定性、结构突变、异常值与覆盖约束。",
    facts: [
      { label: "验证相对 RMSE", value: "2.10%" },
      { label: "区间覆盖", value: "88.9%" },
    ],
  },
  {
    id: "gate",
    label: "公开门",
    status: "ELIGIBLE",
    tone: "pass",
    title: "L0–L4 全部通过，四步预测已冻结",
    description:
      "两次新鲜子进程重放得到相同输出；预测在私有目标揭示前由代码注册。",
    facts: [
      { label: "证据层", value: "5 / 5 PASS" },
      { label: "新鲜重放", value: "2 / 2" },
    ],
  },
  {
    id: "external",
    label: "外部私测",
    status: "NOT RUN",
    tone: "blocked",
    title: "仍需要真正独立的信任节点",
    description:
      "当前主机不能同时保管私有答案、执行评估并授予自己科学资格。",
    facts: [
      { label: "私测消耗", value: "0 / 1" },
      { label: "科学资格", value: "FALSE" },
    ],
  },
] as const;

export const evidenceLevels = [
  {
    level: "L0",
    title: "可复现性",
    status: "PASS",
    evidence: "新鲜重放 2/2 · 输出哈希一致",
  },
  {
    level: "L1",
    title: "数据与图契约",
    status: "PASS",
    evidence: "公开观测 28 · 训练/验证 19/9",
  },
  {
    level: "L2",
    title: "数学一致性",
    status: "PASS",
    evidence: "尺度误差 4.44e−16 · 递归误差 2.78e−17",
  },
  {
    level: "L3",
    title: "科学可接受性",
    status: "PASS",
    evidence: "相对 RMSE 2.10% · 持久性提升 54.42%",
  },
  {
    level: "L4",
    title: "稳健性与边界",
    status: "PASS",
    evidence: "Bootstrap 100% · 区间相对宽度 6.43%",
  },
] as const;
