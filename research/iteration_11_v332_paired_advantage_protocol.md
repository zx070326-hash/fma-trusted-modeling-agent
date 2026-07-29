# 大迭代 11：V3.3.2 paired bootstrap advantage 协议

冻结日期：2026-07-22（Asia/Shanghai）
运行等级：`RL-DRAFT / Lv1 Manual`
父证据：V3.3.1 evolution hash `3184ad0351f96ebd5cf331d4639a9437d3e24e28e68368f687d0d9ade65e36e9`

## V3.3.1 已冻结失败

V3.3.1 在 16 个全新 seed、64 个 case、36 个 performance-eligible case 上得到：宏观 absolute loss improvement `0.0002622862`，95% bootstrap CI `[-0.0059274424, 0.0059426550]`；2 次 `>0.02` 实质负迁移，一侧 95% 比例上界 `0.1647347330`；终态 `acquisition_candidate_failed_v331`。资源、目标契约、共享锚点、信任收据、动作合法性和 synthetic safety 门均成立。不得回写该结果，不得在同一 seed 上重跑 V3.3.1。

失败包事后分析发现，原信任量

```text
q20(gain_active) - q20(gain_fallback)
```

不是 active 相对 fallback 的稳健优势。它分别压缩两个动作的 bootstrap 分布后再作差，丢失同一 bootstrap 世界内两个动作的相关结构。最大的实质负迁移案例在原量上为 `+0.04991`，但逐成员配对差值的 20% 分位数为 `-0.09960`。

## 方法依据与限制

- Glasserman & Yao (1992), DOI `10.1287/mnsc.38.6.884`：共同随机数用于比较随机系统时，收益取决于相关性、单调性和连续性条件，不是无条件保证。
- Nelson & Matejcik (1995), DOI `10.1287/mnsc.41.12.1935`：共同随机数可用于 best-system 选择与多重比较；其保证依赖明确的统计假设。
- Xie, Frazier & Chick (2016), DOI `10.1287/opre.2016.1480`：仿真优化中可联合利用候选间相关性和共同随机数进行成对采样。

这些来源只支持“比较两个动作时保留配对相关结构”的方法方向。V3.3.2 的 12-member ridge/bootstrap ensemble 不是这些论文中的校准抽样分布，因此不宣称置信区间、正确选择概率或理论最优性。

## V3.3.2 唯一变化

资源账本、两个共享随机锚点、交叉激励 NRMSE、动作目录、最终拟合器、风险门、统计门和 router 全部不变。只把第三动作的优势证据改为：

1. 使用 public pilot 与两个真实 synthetic anchor observations 形成与 V3.3.1 相同的 12-member ensemble；
2. active 与 prefrozen random fallback 在每一个相同 ensemble member 下分别计算 fractional goal-risk reduction；
3. 按 member 逐对作差 `gain_active[m] - gain_fallback[m]`；
4. 使用配对差值的 20% 分位数作为 robust paired advantage；
5. 只有 `cross_excitation_nrmse <= 0.05` 且 `paired_advantage_q20 >= 0.03` 时使用 active，否则执行完全相同的 fallback。

`0.03` 沿用 V3.3.1 已冻结的最小 goal-risk advantage，不根据 V3.3.1 的两个失败案例重新调数值。V3.3.1 失败包只用于确认结构性候选能解释已知反例，不得计入 V3.3.2 通过证据。

## 证据与审计边界

- trust receipt 必须绑定两个 anchor action/observation hashes、ensemble/bootstrap seed、active/fallback action hashes、12 个配对差值的 hash、均值、标准差、正值比例、20% 分位数、阈值和最终选择；
- acquisition 不得读取 hidden mechanism、hidden parameters、private probe、真实 target loss 或 expected route；
- 使用 16 个与 V3.1、V3.1.1、V3.2、V3.3、V3.3.1 均不重叠的全新 seed；
- 仍只允许 exploratory terminal status；不允许 confirmation 或 qualification；
- 若宏观 improvement 下界、机制非退化或负迁移比例门任一失败，保留失败且不得演化 router。
