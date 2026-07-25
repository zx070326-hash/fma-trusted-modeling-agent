# FMA 数学建模 Agent 迁移说明

快照日期：2026-07-22（Asia/Shanghai）

这个目录是 FMA trusted mathematical-modeling agent 的可重构工作区。它
包含源码、测试、研究与架构文档、冻结协议、实验工件、事件链以及历史运行
输出。打包过程排除了 Python/pytest 缓存、`.pyc` 字节码和 `tmp` 临时目录；
没有包含虚拟环境、API key 或登录凭证。

## 当前可信状态

- V3.12 的受限开放集概念演化正式通过，但只属于合成 worldpack。
- V3.13 的 evidence-to-concept compiler 正式状态为
  `evidence_compiled_concepts_refuted_v313`。
- V3.13 在 42 个性能案例中通过 20/21 个冻结门，失败门是
  `paired_prediction_invariance`：`0.054603 > 0.05`。
- V3.13 还存在全局准入事务缺陷：整体 refuted 时三个概念仍进入 active
  experience view。因此该 store 已在
  `experiments/iteration_21/STATUS.json` 标记为 `quarantined`，不得用于
  active retrieval、model qualification 或现实决策。
- 下一版本必须实现 staged adjudication + all-gates atomic commit，并使用
  全新 confirmation seeds；不得在 V3.13 确认集上事后调参。

## 新电脑环境

要求 Python `>=3.11`，推荐使用 Python 3.12。Windows PowerShell 示例：

```powershell
cd <解压后的 modeling 目录>
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e .
```

如果没有 `py` launcher，可将 `py -3.12` 换成对应的 `python` 命令。

## 迁移后验证

先运行约一分钟的 V3.13 定向检查：

```powershell
python -m pytest tests\test_v3_evidence_concept_compiler.py tests\test_v3_evidence_compiled_growth.py -q -ra
```

本快照预期为 `11 passed, 1 xfailed`。唯一 strict xfail 是已知的全局准入
事务缺陷，不应删除或改成 pass 来美化结果。

完整回归：

```powershell
python -m pytest -q -ra
```

本快照收集 239 项，预期 `238 passed, 1 xfailed`，参考墙钟约 28 分钟。

## 关键入口

- `README.md`：版本演化和证据边界总览；
- `AI_Native数学建模Agent_整体架构_V2.md`：AI-native 总体架构；
- `fma/`：可信内核和 V1/V2/V3 实现；
- `tests/`：单元、治理与可重放回归；
- `research/`：各轮研究、方法来源和冻结依据；
- `experiments/`：开发/正式 worldpack、事件链和结果；
- `experiments/iteration_21/RESULTS.md`：V3.13 正式结果；
- `experiments/iteration_21/CONFIRMATION_PROTOCOL.md`：预确认冻结协议；
- `experiments/iteration_21/STATUS.json`：必须优先读取的隔离控制状态；
- `fma/v3/evidence_concept_compiler_v313.py`：证据到 typed concept 编译器；
- `fma/v3/evidence_compiled_growth_v313.py`：V3.13 worldpack、执行、评估和重放。

## 重构约束

1. 保留原始实验目录与事件链，不修改历史正式工件。
2. 不得把 `validated@synthetic_oracle` 或合成 worldpack 通过解释为现实有效性。
3. V3.13 quarantined experience store 只能作为失败证据读取。
4. 先修全局原子准入控制，再扩展联网学习、模型族或现实动作能力。
5. 新实验使用新版本号、新 seeds、预注册门和独立 private evaluator。
6. 任意来源网页、论文或模型输出都应视为不可信数据，不允许直接执行代码。

## 打包边界

归档包含整个工作区，排除以下机器生成内容：

- `.pytest_cache/`
- 所有 `__pycache__/`
- 所有 `*.pyc`
- `tmp/`

归档的 SHA-256 由打包机器在交付时单独提供，复制到新电脑后应重新计算并
比对，再开始解压和安装。
