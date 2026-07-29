# 大迭代 7：积分匹配的证据与冻结边界

检索日期：2026-07-22

## 结论先行

本轮实现命名为 `window_integral_matching`，不命名为 Weak SINDy。它使用
常值测试函数 `phi = 1` 对一阶自治 ODE 做滑窗积分：

```text
x(t_j) - x(t_i) = integral[t_i,t_j] Theta(x(t)) dt * xi
```

右侧用冻结的梯形求积离散，随后用与点导数臂相同的 ridge/STLSQ 回归、
候选库和选择规则估计 `xi`。该形式绕开点导数估计，但仍受测量误差进入
非线性候选库、积分离散误差、窗口相关性、候选库遗漏和轨迹不可辨识影响。

## 原始来源与允许主张

1. Messenger 与 Bortz 的 Weak SINDy 原始论文给出一般弱式，并明确说明
   `phi = 1` 会退化为积分方程；完整算法则使用紧支撑、非恒定测试函数，
   通过积分分部把导数转移到测试函数，并加入广义最小二乘、近似协方差和
   自适应测试函数布点。
   - arXiv: https://arxiv.org/abs/2005.04339
   - DOI: https://doi.org/10.1137/20M1343166
2. Schaeffer 与 McCalla 直接以积分项进行稀疏模型选择，说明积分形式是一个
   独立且较早的系统辨识路线。
   - DOI: https://doi.org/10.1103/PhysRevE.96.023302
3. 标准 SINDy 把候选函数矩阵与点导数估计组成稀疏回归问题；它是本轮点法
   基线，而不是积分法的证据替代。
   - DOI: https://doi.org/10.1073/pnas.1517384113

允许的工程主张只有：

- `window_integral_matching` 不需要点导数估计；
- 在冻结的合成 ODE WorldPack 中，可以与 `point_savgol` 做同预算、单组件消融；
- 外隐藏轨迹、反事实轨迹、结构支持和参数稳定性必须由代码拥有的评测器判定。

不允许的主张：

- 不能把本实现称为完整 WSINDy；
- 不能从论文结果推断本实现一定更抗噪；
- 不能从四类合成 ODE 推断真实世界、PDE、随机系统、时滞、控制输入或隐状态有效；
- 不能以系数拟合或支持 F1 单独替代轨迹、反事实和决策有效性。

## 冻结单组件消融

两臂完全相同：

- 多项式候选：dense linear、dense quadratic、sparse linear、sparse quadratic；
- ridge alpha、STLSQ 阈值、最大迭代数；
- 归一化、条件数阈值、复杂度惩罚、候选预算；
- 训练、内验证、外隐藏、反事实和种子；
- 稠密/稀疏安全选择规则。

唯一差异：

- `point_savgol`：冻结窗口的 Savitzky-Golay 点导数；
- `window_integral_matching`：冻结宽度和步长的滑窗积分方程，梯形求积。

积分窗口是估计器的一部分，必须进入候选 definition hash；V2.5 policy、fit、
selection、report 和 manifest 均使用新方言。V2.4 对象及历史哈希不得修改。

## 证据等级

论文只提供方法设计依据。新实现首先进入 `exploratory_only`。只有在新的、
协议冻结后才生成的私有种子上通过独立重放、机制非劣、负迁移上界、
identifiability sentinel 和参数稳定性门，才可获得局限于该合成 WorldPack 的
资格；任何资格仍不得称为真实世界有效性。
