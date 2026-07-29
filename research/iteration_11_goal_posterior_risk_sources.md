# 大迭代 11：目标后验风险 acquisition 来源冻结

冻结日期：2026-07-22（Asia/Shanghai）
运行等级：`L-DRAFT / Lv1 Manual`
用途：只为 V3.2 合成受控动力系统探索定义 acquisition 候选；不授权现实实验、现实决策或方法能力外推。

## 第一性原理问题

V3.1/V3.1.1 已经证明 Harness 能约束、执行并记录实验动作，但没有证明启发式
`D-opt + disagreement + target proxy - cost - empirical risk` 会降低终端预测损失。V3.1.1 的宏改善均值为正，95% 区间仍跨零，且 18 个 eligible case 中有 6 次实质负迁移。因此本轮只改变 acquisition objective；router、估计器、动作目录、三步 horizon、经验风险门和统计门保持不变。

## 原始来源与可迁移主张

1. Wagenmaker, Simchowitz, Jamieson, *Task-Optimal Exploration in Linear Dynamical Systems*, ICML 2021, PMLR 139:10641–10652。原始 proceedings：<https://proceedings.mlr.press/v139/wagenmaker21a.html>。
   - 可迁移主张：实验设计应围绕指定下游任务的 excess risk，而不是无差别辨识全部参数。
   - 不迁移：该文在线性动力系统上的 instance/task-optimal 样本复杂度结论；V3.2 是非线性特征嵌入上的经验代理。

2. Attia, Alexanderian, Saibaba, *Goal-Oriented Optimal Design of Experiments for Large-Scale Bayesian Linear Inverse Problems*, Inverse Problems 34(9), 2018, DOI `10.1088/1361-6420/aad210`，arXiv `1802.06517`：<https://arxiv.org/abs/1802.06517>。
   - 可迁移主张：A-GOODE 最小化终端 quantity of interest 的后验不确定性，而不是参数本身的后验不确定性。
   - 不迁移：线性高斯/PDE 假设、其理论解释和梯度算法。
   - 可复现检索：2026-07-22 访问 `https://export.arxiv.org/api/query?id_list=1802.06517&max_results=1`，HTTP 200，返回 `http://arxiv.org/abs/1802.06517v2`，updated `2018-06-11T04:40:55Z`。

3. Huang, Guo, Acerbi, Kaski, *Amortized Bayesian Experimental Design for Decision-Making*, NeurIPS 2024，DOI `10.52202/079017-3475`。原始 proceedings：<https://proceedings.neurips.cc/paper_files/paper/2024/hash/c59f05d7ab3638b138cc61f32e1a7cd1-Abstract-Conference.html>。
   - 可迁移主张：最大化参数信息不等于最大化下游决策效用，实验与终端决策目标应共同定义。
   - 不迁移：TNDP、摊销神经策略或其任务实验结果。

## 预冻结 V3.2 acquisition

令 `X_t` 是当前公开 pilot 和已执行实验构成的、按当前列尺度归一化的二次多项式特征矩阵。保持 V3.1 的 ridge 参数 `alpha`，定义未校准的参数协方差代理：

```text
Lambda_t = alpha I + X_t' X_t
C_t      = inverse(Lambda_t)
```

对 bootstrap ensemble 成员 `b`，只用公开 problem contract、公开初始状态/边界、公开动作目录和该成员模型构造终端目标特征 `G_b`：

- `controlled_response_prediction`：从公开初始状态运行完整公开动作目录；
- `free_run_prediction`：从预冻结公开尺度 `[0.75, 1.0, 1.25]` 生成初值，运行零输入；非零初值分量按尺度相乘，恰为零的分量按 `(scale-1) * 0.15 * envelope_width` 扰动，随后机械裁剪到距离 envelope 边界 5% 的内部。

候选动作 `a` 在成员 `b` 下产生预测特征 `Z_ab`。定义：

```text
C_ab      = inverse(Lambda_t + Z_ab' Z_ab)
r_b       = trace(G_b C_t G_b') / rows(G_b)
r_ab      = trace(G_b C_ab G_b') / rows(G_b)
g_ab      = (r_b - r_ab) / max(r_b, 1e-12)
score(a)  = quantile_0.20({g_ab across 12 members})
```

实现使用与上述逆矩阵严格对应的白化坐标和 SVD：先构造 `B B' = C_t`，再令
`W = Z_ab B`，在 `W` 的右奇异向量基上以 `1/(1+s_i^2)` 缩放目标特征。
这是为避免病态非线性预测特征使直接求逆丢失正定性的数值实现，不改变目标函数或新增调参。

Harness 在通过既有经验风险硬门的动作中最大化 `score(a)`。动作成本、峰值、能量和切换在目录中相同，因此不再用任意权重重复加减。D-opt、ensemble disagreement、当前/预测后验目标风险、均值/保守风险下降和离散度全部单独写入 receipt；D-opt 和 disagreement 只作诊断，不参与排序。

## 防泄漏与证据边界

- acquisition 不得读取 private probe 初值、private probe 输入、隐藏机制、隐藏参数、隐藏轨迹、真实性能损失或 expected route。
- private pack 只在 spec、两臂 policy 和 acquisition 数学定义封存后生成。
- `C_t` 是 ridge/Laplace 风格协方差代理，不是校准 Bayesian posterior；bootstrap 20% 分位不是覆盖保证。
- V3.2 只裁决 acquisition 是否值得进入下一轮 router 演化。router 保持 V3.1.1 不变，因此即使 acquisition 门通过，也不得产生整体 controlled-epistemic qualification。
- 探索与确认 seed 不重用。只有探索 acquisition 门通过、随后独立 router 版本也通过，才允许冻结一次性 confirmation。
