# 大迭代 6：动力系统方法证据与实现边界

访问日期：2026-07-22。检索是定向检索而非系统综述；来源元数据由 Crossref 单 DOI endpoint 冻结，正文判断优先参考原论文/开放全文。所有网络响应在 Harness 中标为 `untrusted_bibliographic_data`，只能形成待隐藏评测的候选知识。

## 可复现检索

| 用途 | 标识符 | Crossref endpoint | 主文入口 |
|---|---|---|---|
| 稀疏非线性动力学发现 | DOI `10.1073/pnas.1517384113` | `https://api.crossref.org/works/10.1073%2Fpnas.1517384113` | [PNAS/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4839439/) |
| 噪声下弱形式发现 | DOI `10.1137/20M1343166` | `https://api.crossref.org/works/10.1137%2F20m1343166` | [SIAM DOI](https://doi.org/10.1137/20M1343166) |
| 可观测性与结构可辨识性 | DOI `10.1155/2019/8497093` | `https://api.crossref.org/works/10.1155%2F2019%2F8497093` | [开放全文](https://doi.org/10.1155/2019/8497093) |

Crossref 定向返回的标题分别为：

1. *Discovering governing equations from data by sparse identification of nonlinear dynamical systems*；
2. *Weak SINDy: Galerkin-Based Data-Driven Model Selection*；
3. *Observability and Structural Identifiability of Nonlinear Biological Systems*。

## 从来源提出、但尚未直接相信的规则

### 1. 候选库与稀疏性必须显式

SINDy 的核心不是“自动找到任意真方程”，而是在冻结的候选函数库中寻找稀疏组合。因此实现把多项式基、阶数、阈值和回归器写入 exact policy；如果真动力学在所选基中不稀疏，稀疏先验不能保证成功。

### 2. 导数估计是模型的一部分

Weak SINDy 的方法动机之一是避免对含噪数据直接做点导数估计。本轮只实现了显式 Savitzky–Golay 导数估计，并没有实现 weak-form 积分，因此不能把本实现称为 Weak SINDy；该差距保留为下一候选组件。

### 3. 拟合轨迹不等于参数可辨识

可观测性关心能否从输出恢复内部状态；结构可辨识性关心是否能从输出理论上唯一确定参数。本轮代码只有“冻结轨迹 × 冻结候选库”的归一化设计矩阵秩/条件数诊断。它能拒绝部分观测和未激发的合成 sentinel，但不是符号结构可辨识性证明。

### 4. 结构、轨迹和反事实必须分开评分

本轮隐藏 evaluator 分别测：

- private-outer 轨迹 NRMSE；
- 新初值反事实轨迹 NRMSE；
- 隐藏真方程支持集 F1；
- 不可辨识 sentinel 的错误晋级；
- 同候选/拟合预算下的负迁移。

实验结果证明这些指标不可互换：稀疏策略可以显著改善结构 F1，同时让轨迹外推或反事实变差。

## 证据等级

- 来源身份与本地快照完整性：已机械验证。
- 上述三条方法解释：候选解释，受原论文支持但未做逐句语义证明。
- V2.4 实现忠于完整 SINDy/Weak SINDy：未证明；实际只是一个受限多项式 ODE adapter。
- 真实动力系统或参数推断有效性：没有证据。
