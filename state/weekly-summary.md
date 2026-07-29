# FMA 周摘要

## 2026-07-22

项目从“合成优化可信内核”完成首个经验建模大迭代：结构化时间序列、四候选谱、滚动持出、历史残差区间、容量 regret/行动稳定性、弃权、内容寻址与全链复算已落地。稳定合成场景获得有限的 shadow eligibility；结构突变正确弃权；全量 86 测试通过。整体目标继续进行，下一迭代转向至少两个真实公开任务家族、OOD/敏感性和带置信界的 ReleaseGate，仍不开放现实动作。

大迭代 2 已把闭环推到四个官方真实 series/site。默认骨架在 BLS/USGS 均失败；由失败签名冻结的局部窗口与指数平滑在未见 BLS 序列上产生正的配对 bootstrap 改善区间，但未跨到水文，且漂移门阻止总晋级。一次 schema hash 污染被历史重放发现并通过 V2.2 方言修复。最终 95 测试通过，四个有效真实运行全链复算通过；系统仍为 retrospective shadow，不具现实决策资格。

大迭代 3 把长期记忆变成事件溯源认识图，并完成实际网页方法快照、官方运行导入、模拟撤销 cascade 和 private-outer 同预算消融。memory policy 在 12 case 中 4 胜/8 平/0 负迁移且平均改善 95% CI 为正，但因未达到事前 50% strict-win 门保持 `candidate_rejected`。组合裁决与单组件归因已分离；最终 108 测试通过。下一轮只在全新 seed 上验证重新预注册的效用/非劣/负迁移置信门。

大迭代 4 在 80 个全新 case 上确认 memory 的宏平均改善与逐机制非劣，但 1 个 >5% 负迁移使其比例上界 5.79% 超过 5% 安全线，保持拒绝。大迭代 5 从该失败演化 exact policy：保留 global/local trend、退役 SES 槽位；第二组 80 个全新 case 得到 11.97% 宏平均改善、0 个实质负迁移且上界 3.68%，首次获得仅限 synthetic forecast WorldPack 的 qualification。下一阶段转向第二个结构不同的建模家族。

大迭代 6 已建立第二个 Dynamics IR 方言、三 DOI 候选知识链、经验可辨识性弃权、独立积分和四机制 private WorldPack。三轮递进实验显示稀疏 memory 的结构 F1 持续改善，但轨迹/反事实负迁移无法通过预冻结安全门；两个确认 policy 均被 refute，没有生成 Dynamics qualification。这把下一瓶颈定位到 weak/integral matching、参数不确定性、长期稳定性和主动激励，而不是更多 prompt 或更多 agent。

大迭代 7 完成 point-Savitzky–Golay 与 window-integral matching 的同预算单组件消融，并加入 200 次 moving-block 参数稳定性门。15 点窗灾难性失败，41 点失败演化在探索集跨零，80-case 确认则宏改善 -31.08%、39 次实质负迁移，尽管结构 F1 仍提高 6.15 个百分点；integral 23/80 case 也未过稳定性门。exact policy 被 refute、无 qualification；下一瓶颈转为主动激励和实验信息设计。

大迭代 8 把“下一次采什么数据”升级成 Harness 拥有的顺序动作循环：模型只能在冻结的安全初值目录中提出动作，Harness 掌握预算、合法性、隐藏执行、重放和晋级。首轮 32-case 暴露近零基线进入相对效应分母的评测失真；只版本化效应量后，第二组 32-case 探索为正。80-case 确认得到联合损失绝对改善 0.03430（95% CI `[0.01580, 0.05496]`）、5 次实际负迁移（上界 12.69%）、四机制均非劣且宏条件数改善为正，获得仅限 `synthetic_safe_initial_condition_design_v261` 的 qualification。改善主要由 Lotka–Volterra 贡献；连续控制、现实安全与结构可辨识性仍未解决。

大迭代 9 将 Agent 从“冻结问题后的验证流水线”推进为首个受治理认识闭环：稳定使命冻结，episode 问题契约不可变但可由权威证据产生可追溯子版本；采数据和澄清问题成为同一权限/成本框架内的认识动作。一步 V3.0 因澄清挤占数据预算出现 2 次实质负迁移并被保留为失败；只把 horizon 演化为两步后，80-case 全新确认取得 bounded regret 改善 0.09770（95% CI `[0.07494, 0.12175]`）、四机制均正、0 次实质负迁移且零误重构，获得仅限 synthetic sequential capacity problem-reformulation 的 V3.0.1 qualification。它仍不证明开放问题发现、真实有效性或广义数学建模。

大迭代 10 把认识动作接到已知 actuator 的分段常值输入，建立等峰值/能量/切换实验 IR、目标导向 acquisition receipt、经验风险权限门、代码级弃权及 problem/data/model 路由。V3.1 两步探索因澄清后二维系统只剩一次激励而宏效应为负；只将 horizon 2→3 后，V3.1.1 宏均值转正且阻尼振子明显改善，但 CI 仍跨零、6/18 负迁移未消失，Duffing 模型错配路由也失败。两个版本均无 qualification；结果把下一瓶颈收窄到“终端目标导向 acquisition”和独立模型错配诊断，而不是更多算法列表或多 Agent。

大迭代 19 在 V3.10 结构骨架基础上移除语义变量名 shortcut，建立匿名状态、坐标置换/单位缩放成对表示和 pendulum open-set sentinel。时间恢复后的 fresh confirmation 在 70 个性能 case 上取得 100% 受支持 coverage、topology accuracy、open-set 弃权和成对 topology consistency；0 次材料性负迁移，上界 4.189%，13/13 门通过。一次未来审计时间错误在确认包生成前被发现，推动 verifier 增加哈希之外的因果时间与墙钟门。终态只确认冻结目录的表示不变拓扑选择/拒答，不生成 qualification 或 router；下一阶段转向至少两个算子族的 residual-guided open-set 骨架演化。
