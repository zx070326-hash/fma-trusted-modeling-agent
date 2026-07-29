# 大迭代 14 / V3.7：Applicability State 与 Model Challenge 协议

## 结论先行

V3.7 不再调 V3.6 的 q20 cutoff。它检验一个新的、可证伪的命题：在完全相同的公开观测、拟合预算和 private probe 下，显式记录适用性状态并让三个结构复杂度不同的动力学骨架接受轨迹级留一挑战，是否优于固定二次多项式骨架。

本轮是 `L-DRAFT / Lv1 Manual`。只写本地工件和运行合成 private WorldPack；没有定时任务、现实系统动作、qualification、confirmation 或 task-router 权限。

## 检索合同与来源

- 目的：模型判别、模型失配和基于留出预测的模型组合/选择。
- 范围：定向检索，不是系统综述。
- 数据库：OpenAlex Works API；随后核对出版方原始页面。
- 访问日：2026-07-22。
- 查询：
  - `Design of experiments for discriminating between two rival models`
  - `Bayesian calibration of computer models`
  - `Using stacking to average Bayesian predictive distributions`
- 主要记录：
  - Atkinson & Fedorov, 1975, DOI `10.1093/biomet/62.1.57`：竞争模型可以成为实验设计的直接判别目标；原文处理回归模型，不自动覆盖本项目的非线性 ODE。
  - Kennedy & O'Hagan, 2001, DOI `10.1111/1467-9868.00294`：参数不确定性和模型不足是不同来源；本轮不实现其 GP discrepancy，也不继承其 Bayesian 保证。
  - Yao et al., 2018, DOI `10.1214/17-BA1091`：模型组合应以留一预测性能为目标；本轮只借鉴 held-out predictive adjudication，不实现 Bayesian stacking。
  - Plumlee, 2019, DOI `10.1111/rssb.12314`：适用输入区域和模型 discrepancy 的边界必须显式；其置信/一致性结论依赖本轮不满足的假设，因此不迁移保证。

可复现 OpenAlex 端点形式：

```text
GET https://api.openalex.org/works?search.exact=<URL-ENCODED-TITLE>&per_page=5&select=id,doi,title,publication_year,cited_by_count,primary_location,open_access
```

## 冻结比较

### 共同输入

- 4 个隐藏机制：exponential、logistic、damped oscillator、Duffing oscillator；
- 16 个全新 seed，每个机制 16 case，共 64 case；
- 每例只向两个 arm 暴露同一个 pilot 和 action catalog 中预冻结索引 `[0, 7]` 的两条受控轨迹；
- 同一个 Savitzky–Golay 导数、ridge/STLSQ、噪声、actuator、状态边界和 private probe；
- `sensor_calibration_failed` case 必须在两个 arm 同时弃权，不得计入效果分母。

### 两个 arm

1. `fixed_quadratic_baseline`：固定二次多项式 ODE；
2. `applicability_challenge_candidate`：在下列三个 family 中由代码选择：
   - `linear_state_space`：总次数 1；
   - `quadratic_interaction_ode`：总次数 2；
   - `cubic_sparse_ode`：总次数 3。

这只是最小结构 portfolio。三者仍属于多项式 ODE 超家族，因此即使通过，也只能授权下一次真正非嵌套 skeleton 实验，不能声称已经建立通用异构模型库。

## ApplicabilityStateV37

每个 state 只能由公开/已授权观测计算，并绑定：

- case、contract、actuator、envelope 和 observation hashes；
- state dimension 与 decision target；
- quality flags；
- 观测轨迹数和点数；
- 状态空间相对 envelope 的逐维覆盖率；
- 输入幅度覆盖率和输入设计 rank ratio；
- 每个 family 的 challenge hash；
- family 间预测分歧；
- `private_mechanism_seen=false`、`private_probe_seen=false`、`private_target_loss_seen=false`。

缺少上述任一绑定，或者出现 private 字段，整个 case fail closed。

## FamilyChallengeReceiptV37

对三个 family 分别执行轨迹级 leave-one-trajectory-out：每折用两条轨迹拟合，再从被留出轨迹的初态和输入重放完整轨迹。收据记录：

- basis 宽度、normalized rank/rank ratio、condition number；
- 全数据 derivative residual；
- 3 个 held-out trajectory NRMSE、均值、标准误和 simulation failure count；
- 来源 observation hashes；
- 预冻结 eligibility：无 simulation failure、rank ratio `>=0.95`、condition `<=1e8`、mean loss `<=0.35`。

候选 arm 先找最小 held-out mean loss，再使用预冻结 one-standard-error 规则：在 `best_mean + best_standard_error` 内选择 basis 最小的 eligible family。无 eligible family 时必须 `NEEDS_EVIDENCE`。

## Private adjudication

隐藏 evaluator 才能看到真实 mechanism 和 probe trajectories。预冻结最小正确 family：

- exponential、damped → linear；
- logistic → quadratic；
- Duffing → cubic。

报告必须重算：

- 按 seed 聚类 bootstrap 的 macro target-loss improvement 与 95% CI；
- 四机制平均 improvement 和 `>= -0.02` 非劣；
- `improvement < -0.02` 的实质负迁移数及一侧 Clopper–Pearson 95% 上界 `<=0.10`；
- candidate 最大 target loss 不高于 baseline；
- family routing accuracy `>=0.75`；
- candidate 在所有数据质量通过 case 上的模型覆盖率 `>=0.90`；未选择 case 的 private loss 固定按 `10.0` 计入，不得从效果分母消失；
- 两臂公开 context、observation hashes 和挑战收据逐例一致；
- 事件链、内容寻址、private-pack 再生成、bundle 重放和报告复算全部一致。

只有全部通过时，状态才可以是 `model_challenge_ready_for_non_nested_extension_v37`。该状态仍明确：

- `router_experiment_permitted=false`；
- `overall_qualification_permitted=false`；
- `confirmation_permitted=false`；
- `real_world_authorization_permitted=false`。

## 失败纪律

如果 V3.7 失败：

1. 保存完整失败工件；
2. 不事后改 seed、阈值或 private 分母；
3. 只在失败集中显示单一、可解释的结构问题时，冻结 V3.7.1 单组件演化；
4. 若问题来自 portfolio 同质性，则下一步加入非嵌套 family，而不是继续调 one-standard-error 门。
