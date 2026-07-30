# FMA V7 迁移说明

本文用于在新电脑重建公开源码版 FMA。默认分支包含运行所需源码、前端、
核心协议文档和精简契约测试，不包含本机凭证、SQLite 账本、任务工作区、
完整 campaign 输出或迭代研究日志。

## 1. 克隆

```powershell
git clone https://github.com/zx070326-hash/fma-trusted-modeling-agent.git
Set-Location fma-trusted-modeling-agent
git status
```

要求：

- Python 3.11 或更高版本；
- 完整阶段角色需要可用的 Codex CLI；
- Web 前端需要 Node.js 22.13 或更高版本；
- authority key、API 凭证和登录状态必须在仓库外单独迁移或重建。

## 2. Python 环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ".[test]"
```

验证安装和公开契约：

```powershell
python -c "import fma; print(fma.__version__)"
fma-ops --help
fma-studio --help
python -m pytest
```

## 3. 前端

```powershell
Set-Location frontend
npm install
npm test
npm run dev
```

前端通常运行在 `http://localhost:3001`。返回仓库根目录后再启动本地
Studio Bridge。

## 4. 本地 authority material

以下示例在用户目录生成 32 字节本地开发 key，而不是在仓库中生成：

```powershell
$authorityPath = Join-Path $env:USERPROFILE ".fma-authority.key"
$authorityBytes = New-Object byte[] 32
$rng = [Security.Cryptography.RandomNumberGenerator]::Create()
$rng.GetBytes($authorityBytes)
$rng.Dispose()
[IO.File]::WriteAllBytes($authorityPath, $authorityBytes)

$env:FMA_STUDIO_TOKEN = [guid]::NewGuid().ToString("N")

fma-studio `
  --task-root .\tasks `
  --authority-key-file $authorityPath `
  --host 127.0.0.1 `
  --port 8765
```

本地 key 只能建立本地工作流真实性，不能冒充外部 Custodian、Evaluator、
Promotion Authority、KMS/HSM 或独立科学资格节点。

## 5. 迁移后检查

```powershell
fma-ops --task-root .\tasks doctor
git status --short
```

正常运行产生的以下内容已经被 Git 忽略：

- `.venv/`
- `tasks/`
- `.fma-op-v70/`
- `runs/`、`artifacts/`
- `experiments/`、`research/`、`state/`
- `.env*`、私钥和本地签名材料

不要把旧电脑上的任务数据库直接覆盖到正在运行的新实例。先停止进程，
完整复制对应 task root 与其 `.fma-op-v70` 目录，再运行 `doctor`；任何
manifest 或事件链不一致都应保持失败关闭。

## 6. 历史证据边界

公开默认分支为便于阅读的源码发布面。清理前的 V6–V7 campaign、receipts
和研究过程材料保留在 Git tag `evidence-archive-v7.0`，不属于默认安装，
也不应被解释为已经取得外部科学资格。
