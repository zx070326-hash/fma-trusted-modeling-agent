# 大迭代 21：Evidence-to-Concept Compiler 研究与实验设计

检索日期：2026-07-22

检索性质：针对性 primary-source / normative-standard / official-software 调研，不是系统综述。

## 为什么 V3.12 仍不是自主知识学习

V3.12 已确认 `residual -> typed candidates -> public challenge -> private admission` 可以在两个合成 open-set 算子族上工作，但候选 `logarithmic_rate`、`periodic_restoring_force` 和 decoys 都写在 Python 分支里。它证明的是 baseline 目录外、developer-frozen candidate grammar 内的恢复，不是 Agent 从网上阅读、抽取、编译并迁移数学概念。

第一性原理缺口不是“再加一个搜索算法”，而是一个不把不可信文本直接升级为代码权限的知识编译边界：

```text
web source
-> immutable source record
-> supported scientific claim
-> untrusted concept draft
-> typed operator AST
-> unit/domain/static checks
-> deterministic compiler (no eval/exec)
-> public numeric canaries + influence challenge
-> private cross-task adjudication
-> versioned experience store
-> admit / reject / revoke
```

## 可迁移的方法原则

### 1. LLM-SR：骨架生成与常数优化分离

- Primary source: Shojaee et al., *LLM-SR: Scientific Equation Discovery via Programming with Large Language Models*, arXiv:2404.18400v3, ICLR 2025 Oral. <https://arxiv.org/abs/2404.18400>
- 论文把 equation 视为由数学算子组成的 program；LLM 迭代提出 skeleton，再用数据优化参数。
- 本地迁移：Generator 只能提交 declarative AST 和 parameter declarations；常数拟合由 Harness 完成。
- 不迁移：论文 benchmark/OOD 效果不证明来源抽取正确，也不授权执行 LLM 生成的 Python。

### 2. LaSR：concept library 与普通 hypothesis population 分离

- Primary source: Grayeli et al., *Symbolic Regression with a Learned Concept Library*, NeurIPS 2024, DOI 10.52202/079017-1419. <https://proceedings.neurips.cc/paper_files/paper/2024/hash/4ec3ddc465c6d650c9c419fb91f1c00a-Abstract-Conference.html>
- LaSR 在高表现 hypothesis 上抽象/演化 textual concepts，再用 concept-conditioned steps 生成新 hypotheses。
- 本地迁移：跨任务 Experience Store 保存 concept version、source claims、支持/反例、admission 与 revocation，而不是只保存最佳公式。
- 不迁移：文本 concept 本身不是可执行或可信的数学对象。

### 3. IGSR：从单一总分转向 term-level influence

- Primary source: Saveliev et al., *Influence-Guided Symbolic Regression*, arXiv:2605.29184v1, 2026 preprint. <https://arxiv.org/abs/2605.29184>
- 论文让 LLM 生成 candidate basis functions，并用各 term 对泛化精度的边际贡献做 granular feedback/pruning。
- 本地迁移：每个 compiled concept 必须产生 leave-one-term-out influence receipt；总 loss 好但无稳定边际贡献的 term 不进入 warm store。
- 不迁移：这是近期预印本；文中 benchmark 与 wet-lab case 不自动成为本地真实性证据。

### 4. PySR：算子、嵌套和复杂度必须受限

- Official software/docs: PySR repository and operator/options documentation. <https://github.com/MilesCranmer/PySR> and <https://ai.damtp.cam.ac.uk/pysr/options/>
- PySR 显式配置 unary/binary operators、argument complexity、nested constraints、operator complexity，并在 accuracy-complexity 路径上选择。
- 本地迁移：固定 operator whitelist、AST node/depth budget、operator-specific domain rules 和 complexity；来源不能自定义运行时代码。
- 不迁移：PySR 允许 Julia custom operator；FMA 在本轮禁止这一路径，因为它扩大到任意代码执行。

### 5. PROV-O：生成、派生、归属和失效必须显式

- Normative standard: W3C PROV-O Recommendation. <https://www.w3.org/TR/prov-o/>
- PROV-O 提供 Entity、Activity、Agent 以及 derivation/generation/invalidation 等 provenance 关系。
- 本地迁移：SourceRecord、ScientificClaim、ConceptDraft、CompiledConcept、Attempt、Admission 和 Revocation 都内容寻址并显式绑定上游 hash。
- 不迁移：采用 provenance vocabulary 不能证明来源真、抽取忠实或模型因果正确。

### 6. SBML：数学交换需要明确结构与单位语义

- Normative specification: *SBML Level 3 Version 2 Core Release 2* (2019). <https://sbml.org/documents/specifications/level-3/version-2/core/>
- SBML 使用受规范约束的模型对象与 MathML，并提供 time/species/parameter 等单位语义。
- 本地迁移：每个 state、time、parameter 和 AST 子表达式都带 dimension vector；`add/sub` 必须同维、`log/exp` 输入必须无量纲、最终 RHS 必须是 state/time。
- 不迁移：V3.13 不声称兼容完整 SBML，也不把 XML 合法性等同于科学有效性。

## 概念来源记录

本轮只把下列经典机制当作**合成测试概念来源**；论文/元数据支持其历史和函数家族，WorldPack 的具体参数与 OOD probes 仍由 Harness 私有生成：

- F. J. Richards, *A Flexible Growth Function for Empirical Use*, Journal of Experimental Botany 10(2), 290–301 (1959), DOI `10.1093/jxb/10.2.290`. Bibliographic record: <https://cir.nii.ac.jp/crid/1360855569998926848?lang=en>
- Gaudy et al., *Control of Growth Rate by Initial Substrate Concentration at Values Below Maximum Rate*, Applied Microbiology 22(6), 1041–1047 (1971), DOI `10.1128/am.22.6.1041-1047.1971`. The paper explicitly tests the Monod hyperbolic relationship and also reports a failure of its instantaneous interpretation; that negative result must remain in the claim record. <https://journals.asm.org/doi/10.1128/am.22.6.1041-1047.1971>
- Gompertz growth is carried forward from V3.12 only as a previously privately admitted synthetic concept, not as a newly source-approved real mechanism.

## V3.13 typed concept contract

### Cold evidence zone

`SourceRecordV313`

- title, persistent ID, canonical URL, source tier, retrieval date;
- bibliographic/record hash, not a falsely claimed full-page snapshot hash;
- known support and known contradiction/limitation summaries;
- no execution permission.

`ScientificClaimV313`

- source record hash and human-readable locator;
- claim kind: operator template, parameter role, unit relation, scope condition, limitation;
- paraphrased claim; extraction is marked `sol_research_draft`, not authoritative;
- claim hash and optional contradiction links.

### Warm candidate zone

`ConceptPackageV313`

- concept ID/version, source claim hashes;
- recursive declarative AST using only `state`, `parameter`, `constant`, `add`, `subtract`, `multiply`, `divide`, `log`, `power`;
- parameter bounds and dimension vectors;
- state/time dimensions, domain preconditions, node/depth/parameter budgets;
- no Python/Julia/SymPy strings and no custom callable.

`CompiledConceptV313`

- compiler version/source hash;
- static type/unit proof receipt for every node;
- numeric domain-canary receipt;
- package hash and compiled hash;
- callable exists only as Harness interpretation of the AST, never as source-provided code.

### Hot run state

- residual/statistical signature for the current anonymous case;
- equal-budget attempts and fitted parameters;
- validation/challenge loss, complexity and leave-one-term-out influence;
- stop reason, invalid-domain events and all failed attempts.

### Cross-task Experience Store

- immutable events: proposed, compiled, rejected_static, rejected_numeric, development_supported, privately_admitted, contradicted, revoked;
- a concept version is active only if its latest valid event is `privately_admitted` and no later revocation binds it;
- public score and LLM confidence cannot write admission events;
- private feedback updates ledger status but is not returned to the Generator.

## 拟议 V3.13 析因实验

### 目的

检验外置 source-grounded concept packages 能否经过同一个通用 compiler，在**全部为一维匿名状态**的条件下区分三个可混淆机制；避免 V3.12 用 state dimension 暗示 operator family。

### 三个一维隐藏机制

1. Gompertz capacity growth：`r*x*log(K/x)`；
2. Richards generalized growth：`r*x*(1-(x/K)^nu)`；
3. Monod-like net saturation：`a*x/(b+x)-c*x`。

所有公共 case 只暴露 `z0(t)`、四条轨迹与质量标志。mechanism、真实 concept、参数、scale representation、pair ID 和三条 private OOD probes 均私有。

### 两臂

- baseline：四个固定多项式 degree candidates；
- candidate：同一组四个 externally serialized concept packages（三个目标 + 一个单位合法但机理错误的 decoy），由通用 AST compiler 执行。

两臂每 case 各四次表达式评估；相同 trajectory、导数、常数优化、validation/challenge 和 private probes。candidate 不按维度路由，因为所有 case 都是一维且收到同一候选集合。

### 必须失败的反例

- `log(state)` 未先除以同单位 scale：unit checker 拒绝；
- `add` 两侧单位不同：拒绝；
- `power` exponent 有单位：拒绝；
- source hash/claim hash 不存在：拒绝；
- AST 超 node/depth budget：拒绝；
- divide canary 接近零或产生非有限值：numeric sandbox 拒绝；
- source 提供 Python/Julia/custom operator 字段：schema 拒绝；
- 公共成绩直接写 `admitted`：ledger 拒绝；
- 后续反例绑定 concept version：active view 撤销但历史事件不删除。

### 正式门草案

- source/claim/package/compiler hash chain complete = 100%；
- static-invalid adversaries rejected = 100%；
- numeric-invalid adversaries rejected = 100%；
- all attempts persisted, equal budget, no arbitrary code/private feedback = true；
- candidate coverage、exact concept recovery、representation pair consistency ≥ 0.90；
- decoy selection = 0；
- maximum paired private loss difference ≤ 0.05；
- bootstrap improvement lower 95% > 0；
- mechanism/representation non-regression；
- material negative-transfer upper 95% ≤ 0.10；
- admitted concepts active, contradicted version revoked, public-only candidate never active；
- no model qualification、task router 或 real-world action。

## 不可声称边界

即使 V3.13 全过，也只说明：Sol 从本轮检索到的来源整理出的 declarative concept drafts，可以被安全 compiler 和私有合成 WorldPack 审查、迁移和撤销。它仍不证明 LLM 自动抽取忠于任意论文，不证明论文机制适用于现实数据，不证明单位/变量映射已自动解决，也不批准无人复核的现实科学结论。
