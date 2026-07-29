# 大迭代 18：V3.10 状态拓扑骨架因子实验协议

状态：`FROZEN_BEFORE_NEW_PRIVATE_WORLDPACK`

冻结日期：2026-07-22

## 1. 决策问题

V3.9.1 已恢复验证器的6段输入语义，但 generic polynomial portfolio 在 Duffing 私有探针上的均值损失仍为 `1.10359`。本迭代不再问“多项式再升几次能否拟合”，而是区分三类原因：

1. **骨架**：状态拓扑和方程约束是否错误；
2. **估计器**：显式数值微分是否放大噪声；
3. **验证法**：跨实验 LOO 是否遗漏时间外推失败。

## 2. 外部方法证据及转移边界

检索范围是有目的、非穷尽的一手论文和官方文档。外部内容只作为候选方法来源，不转移任何正确性保证。

- Brunton, Proctor & Kutz (2016), *Discovering governing equations from data by sparse identification of nonlinear dynamical systems*, DOI `10.1073/pnas.1517384113`：借用显式候选函数库和稀疏动力学比较原则。
- Messenger & Bortz (2021), *Weak SINDy: Galerkin-Based Data-Driven Model Selection*, DOI `10.1137/20M1343166`：借用积分弱形式避免直接估计导数的原则。
- scikit-learn `TimeSeriesSplit` 官方文档：借用保持时间顺序、未来块只作验证的原则。

## 3. 开发证据与正式证据隔离

代码和选择规则只在 V3.9.1 已经打开过的前12个seed上做开发诊断。这一诊断不能进入V3.10的正式科学结论。

开发诊断曾发现两类问题，并在新私有包生成前修复：

1. 当前 NumPy 已移除 `trapz` 别名，积分算子固定为 `numpy.trapezoid`；
2. 单凭简洁性会在公共证据统计等价时强制切换。最终规则增加公开LOO切换门：候选LOO均值加一标准误必须优于基线LOO均值，否则保留基线。

正式证据只允许来自固定的新seed：

`25013, 25073, 25127, 25183, 25247, 25309, 25367, 25423, 25481, 25537, 25589, 25643`

共48 case，其中39个公共质量合格case、9个由公共 `pilot.quality_flags` 决定的质量弃权case。

## 4. 冻结因子

### 骨架

- `generic_polynomial_degree_1`
- `generic_polynomial_degree_2`
- `generic_polynomial_degree_3`
- `first_order_rate_law`：一维状态，候选项为 `x, x²`
- `second_order_kinematic_force_law`：二维 `position, velocity`，硬约束 `position' = velocity`，只估计 `velocity'` 中的 `position, velocity, position³`

后两者通过公开状态维数和状态名称判定兼容性，不能读取私有 mechanism 标签。

### 估计器

- `pointwise_savgol_stlsq`
- `integral_trapezoid_ridge`，4个时间间隔滑动积分窗

### 验证法

- `trajectory_leave_one_experiment_out`
- `blocked_tail_forecast`，只用每条轨迹前31点拟合，从第31点向未来预测

每个质量合格case有4个兼容骨架、2个估计器、2个验证器，共16个cell、8个骨架-估计器pair。每个验证cell有3个正确绑定到6段输入的仿真fold，正式包应有 `39 × 16 × 3 = 1872` 个有效输入绑定。

## 5. 两条策略

基线：只使用1/2/3次嵌套多项式、Savitzky-Golay导数和跨实验LOO，以one-standard-error规则选最低次数。

候选：要求同一骨架-估计器pair同时通过LOO和blocked-tail，先最小化两者中较坏的分数，再用结构简洁性打破统计等价；最后应用公开LOO切换门。候选不能读取：

- 私有 mechanism；
- 私有 probe；
- 私有 target loss；
- `performance_eligible`；
- 真实世界反馈。

## 6. 预注册门

只有全部门通过，V3.10才授权一次跨域骨架确认实验：

1. 9个质量case均由公共质量标志弃权；
2. 39个性能case的16-cell矩阵完整；
3. 1872个仿真输入绑定全部有效；
4. 生成器与私有评估值隔离；
5. 候选选择覆盖率至少90%；
6. 一阶和二阶结构骨架都至少被公共规则选择一次；
7. 配对改进bootstrap 95% CI下界大于0；
8. Duffing均值改善至少0.2且候选均值损失不高于0.5；
9. 其他机制均值退化不超过0.02；
10. 材料性负迁移的一侧95%上界不高于10%；
11. task router、模型qualification和现实动作全部关闭。

任一门失败，状态必须为 `skeleton_factorial_refuted_v310`。不允许在看到新私有结果后修改阈值、seed、候选库或选择规则；后续修复必须新建版本并使用另一组fresh seeds。

## 7. 能证明与不能证明

通过只能证明：在这一封闭合成动力学方言中，结构骨架、积分估计和双验证选择相对旧嵌套多项式基线具有可重放的增量价值，并授权下一次跨域私有确认。

它不能证明：未知变量发现、PDE/随机/离散事件迁移、现实数据有效性、结构可辨识性、模型qualification、任务自治或现实动作安全。
