# FMA Modeling Studio

FMA 的本地优先 Web 控制台。它展示任务 Intake、S0–S6 图、候选分支、
验证状态、恢复动作、Operator ledger 和 claim ceiling。

浏览器不持有 authority key，也不能自行打开 Gate 或授予科学资格。真实
Codex/FMA 执行通过仅绑定 loopback 的 `fma-studio` Bridge 完成。

## 本地前端

要求 Node.js 22.13 或更高版本：

```powershell
npm install
npm test
npm run dev
```

开发服务器通常位于 `http://localhost:3001`。

## 连接本地 Bridge

先在仓库根目录安装 Python 包：

```powershell
python -m pip install ".[test]"
fma-studio --help
```

启动 Bridge 时必须提供：

- 位于任务工作区之外、至少 32 字节的 authority key；
- 至少 24 字符的 `FMA_STUDIO_TOKEN`；
- 一个本地 task root；
- 如需真实阶段角色，机器上可用的 Codex CLI。

Bridge 只允许绑定 `127.0.0.1`、`::1` 或 `localhost`。默认允许来源是
`http://localhost:3001` 和 `http://127.0.0.1:3001`。

## 能力边界

当前后半链注册了正值标量自治 ODE 和 adaptive positive-series 两个窄域
能力方向。界面中的阶段完成、Operator `ACCEPTED` 和本地测试通过都不等于
外部科学资格、机制真实性或现实行动授权。
