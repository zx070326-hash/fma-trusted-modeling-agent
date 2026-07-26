# FMA Modeling Studio

FMA 的本地优先数学建模工作台。生产站点可查看界面和冻结公开运行；真实 Codex/FMA
执行通过本机 `fma.studio` bridge 完成，authority key 不进入浏览器。

## Local development

```powershell
npm install
npm run dev
npm test
```

开发服务器通常位于 `http://localhost:3001`。连接桥后，当前可执行纵切为：

```text
create task
  → freeze V5 workspace
  → fresh Codex S0 generator
  → typed artifact validation
  → mechanical check
  → fresh independent referee
  → harness-owned S0 gate
  → graph-native S1 candidate frontier and independent review
  → user-supplied positive scalar ODE data freeze
  → registered S2–S6 computation, L0–L4 checks, dossier, and PDF
```

后半链路只支持至少 12 个严格递增时间点的正值标量序列，以及
`scalar_autonomous_ode_v52` 注册候选族。任意模型、文件上传、远程执行、外部私测、
科学资格和现实行动授权尚未接通，界面必须继续按真实状态显示。
