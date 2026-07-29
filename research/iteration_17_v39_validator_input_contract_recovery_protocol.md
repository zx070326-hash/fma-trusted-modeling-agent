# 大迭代 17 / V3.9：Validator Input-Representation Contract Recovery

冻结时间：2026-07-22（fresh private WorldPack 生成前）

## 1. 事故定义

V3.7–V3.8.1 的 held-out simulation 把 `ControlledObservationReceipt.inputs` 的 49 行逐时刻输入直接传入 `_simulate_model_v31`。该模拟器的契约却是 6 段 `PiecewiseConstantInputAction.input_values`，内部按 `floor(t / segment_duration)` 取索引。因此验证器实际只读取逐时刻数组的前 6 行，几乎把第一段控制错误延长到全程。

这个缺陷同时被 generator、evaluator 和 verifier 重放，因而“哈希一致、重放通过”，但科学语义错误。旧工件保留，不修改历史函数；本轮新增方言并显式比较旧/新输入契约。

官方 FMI 3.0.2 把混合 ODE 描述为分段连续系统，并要求输入不连续点作为事件/通信点处理。该来源只支持“输入时间语义必须显式”，不向本系统转移 FMI 合规保证。

## 2. 单变量变化

保持：

- 同一 pilot + catalog action `[0, 7]`；
- linear/quadratic/cubic family；
- Savitzky–Golay 导数、ridge/STLSQ、rank/condition 门；
- 三折 leave-one-observation-out；
- CV loss `<=0.35`、one-standard-error selection；
- private target loss、未解析 loss `10.0`、bootstrap 与负迁移规则。

只改变 held-out simulation 的输入绑定：

- legacy arm：49 行 expanded observation inputs 继续被 segment simulator 误读；
- recovered arm：pilot 绑定 6 段零输入；action observation 必须由 `action_hash` 唯一绑定 public catalog 的 6 段 `input_values`。

每个 fold 保存输入来源、行数、segment 数、action hash、绑定 hash 和是否通过契约。

## 3. Training evidence

在 V3.8.1 的 22 个 acquisition target case 上，只使用 acquisition 前的三条 observation。V3.8.1 的 target-conditioned 两个受控 folds 原报告为 legacy `0/22`；V3.9 为保持原始 V3.7 三折语义，重新包含 pilot fold后得到：

- legacy full-three-fold resolved：8/22；
- recovered generic-polynomial resolved：22/22；
- recovered mean public CV 约 `0.02854`；
- recovered mean private target loss 约 `0.24473`；
- Duffing mean private loss 约 `1.05843`，必须保留为 skeleton/OOD gap，禁止把 validator 修复冒充模型 qualification。

## 4. Fresh experiment

- 16 个与 V3.7–V3.8.1 均不相交的新 seeds × 4 mechanisms；
- 两臂用相同 private pack、相同 observation、相同 family 与 estimator；
- quality-failure case 两臂都必须弃权；
- performance case 中未选模型按 loss `10.0` 计；
- private evaluator 重算两臂 target loss、coverage、逐机制均值、2000 次配对 bootstrap CI 和 material negative transfer 上界。

## 5. 预冻结恢复门

只有全部满足才输出 `validator_input_contract_recovered_ready_for_skeleton_factorial_v39`：

1. legacy 的 action folds 全部记录 expanded-row misuse；
2. recovered 的 action folds 全部由 action hash 唯一绑定 6 段 catalog input；
3. quality abstention 完全一致；
4. recovered coverage 比 legacy 至少高 `0.50`；
5. paired target-loss improvement 的 bootstrap 95% 下界严格大于 0；
6. 至少一个机制的 recovered mean private loss仍大于 `0.20`，使 skeleton gap 可见；
7. 两臂均无 qualification、confirmation、task router 或现实动作授权；
8. fresh pack、两臂、报告、内容寻址和事件链可独立重放。

该终态只恢复 validator 的输入语义，并授权下一次 skeleton/estimator factorial。若任何门失败，输出 `validator_input_contract_recovery_failed_v39`。

## 6. 撤销规则

V3.7–V3.8.1 的内容哈希和机械 verifier 保留；其关于“无 eligible family”“应继续采证”“一次采证无模型 resolved”的科学归因标记为 `superseded_by_validator_input_contract_bug_v39`。V3.8 的 target clarification lineage 结论不依赖该数值模拟缺陷，可继续保留其窄范围。
