export const task = {
  id: "i36-wb-f078f2e7f23214c3ed07",
  shortId: "i36-wb-f078…",
  commit: "5aaecc3",
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
      "来源选择发生在协议提交之后；I34 与 I35 在身份、响应字节和来源记录三个命名空间中都被排除。",
    facts: [
      { label: "先前任务排除", value: "I34 + I35" },
      { label: "抽取次数", value: "1" },
      { label: "来源身份", value: "WITHHELD" },
    ],
  },
  {
    id: "primary",
    label: "自主 ODE",
    status: "L3 FAIL",
    tone: "fail",
    title: "指数趋势模型没有获得科学资格",
    description:
      "初始图选择 exponential.trend_only，但验证误差、创新相关与区间覆盖没有同时满足冻结门槛，因此失败闭合。",
    facts: [
      { label: "候选", value: "exponential.trend_only" },
      { label: "验证相对 RMSE", value: "18.43%" },
      { label: "区间覆盖", value: "11.1%" },
    ],
  },
  {
    id: "router",
    label: "表征路由",
    status: "TRIGGERED",
    tone: "warn",
    title: "失败触发换表征，而不是放宽阈值",
    description:
      "路由器收到 primary_l3_fail，按冻结图生成两个对数增长候选。原阈值、任务和私有目标保持不变。",
    facts: [
      { label: "触发原因", value: "primary_l3_fail" },
      { label: "恢复候选", value: "2" },
      { label: "阈值变化", value: "0" },
    ],
  },
  {
    id: "ar1",
    label: "增长 AR(1)",
    status: "REJECTED",
    tone: "fail",
    title: "误差更低，但参数稳定性不够",
    description:
      "log_growth_ar1 的验证误差最低，但 φ 窗口范围 0.3046 超过冻结上限 0.3000，因此不能因为表现好就越过机制门。",
    facts: [
      { label: "验证相对 RMSE", value: "1.69%" },
      { label: "φ 窗口范围", value: "0.3046" },
      { label: "冻结上限", value: "0.3000" },
    ],
  },
  {
    id: "recovery",
    label: "对数漂移",
    status: "SELECTED",
    tone: "pass",
    title: "对数随机游走漂移通过全部防线",
    description:
      "该模型不是候选中误差最低的，却是唯一同时满足误差、提升、稳定性、结构突变、异常值与覆盖率约束的恢复方向。",
    facts: [
      { label: "验证相对 RMSE", value: "2.10%" },
      { label: "持久性提升", value: "54.42%" },
      { label: "区间覆盖", value: "88.9%" },
    ],
  },
  {
    id: "gate",
    label: "公开门",
    status: "ELIGIBLE",
    tone: "pass",
    title: "L0–L4 全部通过，预测由代码注册",
    description:
      "两次新鲜子进程重放得到相同输出，五层证据全部通过；四个预测在私有目标揭示前冻结。",
    facts: [
      { label: "证据层", value: "5 / 5 PASS" },
      { label: "新鲜重放", value: "2 / 2" },
      { label: "注册预测", value: "4" },
    ],
  },
  {
    id: "external",
    label: "外部私测",
    status: "BLOCKED",
    tone: "blocked",
    title: "还缺一个真正独立的信任节点",
    description:
      "当前主机不能同时保管私有答案、执行评估并授予自己资格。预测已就绪，但外部私测预算仍为 0/1。",
    facts: [
      { label: "外部主机", value: "NOT ESTABLISHED" },
      { label: "私测消耗", value: "0 / 1" },
      { label: "科学资格", value: "FALSE" },
    ],
  },
];

export const candidates = [
  {
    id: "primary",
    family: "PRIMARY · AUTONOMOUS ODE",
    name: "exponential.trend_only",
    status: "FAILED",
    rmse: "18.43%",
    improvement: "26.56%",
    reason: "创新相关、误差与覆盖未能共同过门。",
    selected: false,
  },
  {
    id: "ar1",
    family: "RECOVERY · LOG GROWTH",
    name: "log_growth_ar1",
    status: "REJECTED",
    rmse: "1.69%",
    improvement: "63.45%",
    reason: "φ 窗口 0.3046 > 0.3000；即使误差最低也拒绝。",
    selected: false,
  },
  {
    id: "drift",
    family: "RECOVERY · LOG GROWTH",
    name: "log_random_walk_drift",
    status: "SELECTED",
    rmse: "2.10%",
    improvement: "54.42%",
    reason: "唯一通过全部冻结科学约束的恢复候选。",
    selected: true,
  },
];

export const evidenceLevels = [
  {
    level: "L0",
    title: "可复现性与运行身份",
    description: "两次独立新鲜进程、相同语义输入、相同确定性输出。",
    metrics: ["新鲜重放 2/2", "输出哈希一致", "运行环境已绑定"],
  },
  {
    level: "L1",
    title: "数据与图契约",
    description: "正值有限状态、规则时间步、冻结阈值与完整候选图。",
    metrics: ["公开观测 28", "增长候选 2", "训练/验证 19/9"],
  },
  {
    level: "L2",
    title: "数学一致性",
    description: "对数往返、尺度不变性、递归闭式解与均值回归检查。",
    metrics: ["尺度误差 4.44e−16", "递归误差 2.78e−17", "对数往返误差 0"],
  },
  {
    level: "L3",
    title: "科学可接受性",
    description: "路由正确触发，最终候选通过全部机制与统计防线。",
    metrics: ["相对 RMSE 2.10%", "持久性提升 54.42%", "创新相关 0.257"],
  },
  {
    level: "L4",
    title: "稳健性与声明边界",
    description: "Bootstrap、区间宽度、窗口敏感性和表征恢复收益。",
    metrics: ["Bootstrap 100%", "区间相对宽度 6.43%", "恢复提升 88.59%"],
  },
];

const observations = [
  102.71184, 104.18919, 103.35621, 103.72126, 106.45201, 111.14252,
  115.75142, 116.30113, 120.12768, 122.89709, 123.6341, 128.30736,
  129.00902, 132.79217, 139.57029, 147.81424, 157.3815, 168.16329,
  168.5539, 174.12788, 188.15993, 197.79579, 206.83245, 214.68763,
  225.46519, 236.66567, 246.63762, 256.10347,
];

export const forecastSeries = [
  ...observations.map((value, index) => ({
    value,
    step: index,
    kind: "observed" as const,
  })),
  ...task.predictions.map((value, index) => ({
    value,
    step: observations.length + index,
    kind: "predicted" as const,
  })),
];
