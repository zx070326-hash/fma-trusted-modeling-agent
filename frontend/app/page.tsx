"use client";

import { useState } from "react";
import {
  authorityRoles,
  completedTask,
  evidenceLevels,
  firstPrincipleQuestions,
  graphNodes,
  modelingLoop,
  stages,
} from "./modeling-data";
import { useStudioBridge } from "./use-studio-bridge";

type ViewId = "workspace" | "graph" | "delivery";
type StudioBridge = ReturnType<typeof useStudioBridge>;

const queryTypes = ["自动判断", "解释", "预测", "优化", "控制", "设计"];

function Pill({
  children,
  tone = "neutral",
}: {
  children: React.ReactNode;
  tone?: "neutral" | "blue" | "green" | "amber" | "red";
}) {
  return <span className={`pill pill-${tone}`}>{children}</span>;
}

function StageNavigator({
  selectedStage,
  setSelectedStage,
  completed = false,
}: {
  selectedStage: string;
  setSelectedStage: (stage: string) => void;
  completed?: boolean;
}) {
  return (
    <div className="stage-list" aria-label="S0 到 S6 建模阶段">
      {stages.map((stage, index) => {
        const isActive = selectedStage === stage.id;
        const isLocked = !completed && stage.status === "locked";
        return (
          <button
            className={`stage-item ${isActive ? "stage-selected" : ""} ${
              isLocked ? "stage-locked" : ""
            }`}
            key={stage.id}
            type="button"
            onClick={() => setSelectedStage(stage.id)}
            aria-pressed={isActive}
          >
            <span className="stage-marker">
              {completed ? "✓" : index === 0 ? "●" : index + 1}
            </span>
            <span className="stage-copy">
              <strong>
                {stage.id} · {stage.label}
              </strong>
              <small>{stage.detail}</small>
            </span>
            <span className="stage-owner">{stage.owner}</span>
          </button>
        );
      })}
    </div>
  );
}

function IntakeWorkspace({
  objective,
  setObjective,
  bridge,
}: {
  objective: string;
  setObjective: (value: string) => void;
  bridge: StudioBridge;
}) {
  const [queryType, setQueryType] = useState("自动判断");
  const [files, setFiles] = useState<string[]>([]);
  const [draftCreated, setDraftCreated] = useState(false);
  const canCreate = objective.trim().length >= 12;
  const s0Open = bridge.task?.workflow.stage_statuses.S0 === "gate_open";
  const agentRunning = bridge.task
    ? ["accepted", "running"].includes(bridge.task.activity)
    : false;

  const createTask = async () => {
    if (bridge.connected) {
      try {
        await bridge.createTask(objective);
        setDraftCreated(false);
      } catch {
        return;
      }
    } else {
      setDraftCreated(true);
    }
  };

  return (
    <>
      <section className="work-hero">
        <div className="work-hero-copy">
          <div className="section-kicker">
            <span>ACTIVE STAGE · S0</span>
            <Pill tone="blue">问题定义中</Pill>
          </div>
          <h1>从真实问题开始，<br />不是从模型名称开始。</h1>
          <p>
            描述你真正要理解、预测或优化的现实对象。Agent
            会先澄清边界、决策和错误代价，再组织经典骨架、生成变体与证伪路线。
          </p>
        </div>
        <div className="principle-chain" aria-label="建模主链">
          <span>现实问题</span>
          <i>→</i>
          <span>可证伪任务</span>
          <i>→</i>
          <span>候选竞争</span>
          <i>→</i>
          <span>可信交付</span>
        </div>
      </section>

      <section className="work-card intake-card">
        <div className="card-heading">
          <div>
            <span className="section-kicker plain">01 / TASK INTAKE</span>
            <h2>你现在想解决什么？</h2>
          </div>
          <span className="required-note">问题、数据和价值约束</span>
        </div>

        <label className="objective-field">
          <span>现实问题</span>
          <textarea
            value={objective}
            onChange={(event) => {
              setObjective(event.target.value);
              setDraftCreated(false);
            }}
            placeholder="例如：我们需要预测未来 12 周的急诊到诊量，用于排班；漏配人手的代价高于富余，但不能使用患者级敏感数据……"
            rows={5}
          />
          <small>
            建议包含：对象、时间范围、可用数据、最终决策和最不能接受的错误。
          </small>
        </label>

        <div className="intake-controls">
          <div className="control-block">
            <span className="control-label">问题类型</span>
            <div className="segmented" role="group" aria-label="选择问题类型">
              {queryTypes.map((type) => (
                <button
                  type="button"
                  key={type}
                  className={queryType === type ? "selected" : ""}
                  onClick={() => setQueryType(type)}
                  aria-pressed={queryType === type}
                >
                  {type}
                </button>
              ))}
            </div>
          </div>

          <label className="file-drop">
            <input
              type="file"
              multiple
              onChange={(event) =>
                setFiles(
                  Array.from(event.target.files ?? []).map((file) => file.name),
                )
              }
            />
            <span className="file-plus">+</span>
            <span>
              <strong>{files.length ? `已选择 ${files.length} 个文件` : "添加数据或题面"}</strong>
              <small>
                {files.length
                  ? files.join(" · ")
                  : "CSV、XLSX、PDF、图片或说明文档"}
              </small>
            </span>
          </label>
        </div>

        <div className="intake-footer">
          <div className="truth-note">
            <span aria-hidden="true">i</span>
            文件当前只停留在浏览器；未接入执行服务前不会上传或运行模型。
          </div>
          <button
            className="primary-button"
            type="button"
            disabled={!canCreate || bridge.busy}
            onClick={() => void createTask()}
          >
            {bridge.connected ? "创建真实 FMA 任务" : "建立本地任务草案"}
            <span aria-hidden="true">→</span>
          </button>
        </div>

        {draftCreated && (
          <div className="draft-receipt" role="status">
            <div>
              <Pill tone="green">草案已建立</Pill>
              <strong>{queryType}型建模任务</strong>
            </div>
            <p>
              下一步应由真实 Agent 生成 S0
              问题契约并接受 Harness 检查；本页面没有伪造这次模型调用。
            </p>
            <button type="button" disabled title="等待执行服务 API">
              启动 Agent · 待接入执行服务
            </button>
          </div>
        )}

        {bridge.task && (
          <div className="live-task-receipt" role="status">
            <div className="live-task-head">
              <div>
                <Pill tone={s0Open ? "green" : "blue"}>
                  {s0Open ? "S0 GATE OPEN" : "真实任务已冻结"}
                </Pill>
                <strong>{bridge.task.task_id}</strong>
              </div>
              <button
                className="run-agent-button"
                type="button"
                disabled={agentRunning || bridge.busy || s0Open}
                onClick={() => void bridge.runS0()}
              >
                {s0Open
                  ? "S1 接口待接入"
                  : agentRunning
                    ? "Codex 正在完成 S0…"
                    : "启动 Codex 完成 S0"}
              </button>
            </div>
            <div className="live-event-list">
              {bridge.task.events.slice(-4).map((event) => (
                <div key={event.event_hash}>
                  <span className={`event-state event-${event.status}`} />
                  <span>
                    <strong>{event.message}</strong>
                    <small>
                      #{event.sequence} · {event.event_type}
                    </small>
                  </span>
                </div>
              ))}
            </div>
            <p>
              Graph verified:{" "}
              {bridge.task.workflow.graph_verified ? "true" : "false"} ·
              scientific qualification: false · real-world action: false
            </p>
          </div>
        )}

        {bridge.error && <p className="bridge-error">{bridge.error}</p>}
      </section>

      <section className="work-card diagnostic-card">
        <div className="card-heading">
          <div>
            <span className="section-kicker plain">02 / REGIME DIAGNOSIS</span>
            <h2>Agent 首先要回答的四个问题</h2>
          </div>
          <Pill>对应真实 S0 契约</Pill>
        </div>
        <div className="question-grid">
          {firstPrincipleQuestions.map((question) => (
            <article key={question.id}>
              <span>{question.index}</span>
              <strong>{question.title}</strong>
              <p>{question.description}</p>
            </article>
          ))}
        </div>
      </section>
    </>
  );
}

function LockedStage({
  stageId,
  setSelectedStage,
}: {
  stageId: string;
  setSelectedStage: (stage: string) => void;
}) {
  const stage = stages.find((item) => item.id === stageId)!;
  return (
    <section className="locked-canvas">
      <span className="locked-token">{stage.id}</span>
      <Pill tone="amber">前置 Gate 未打开</Pill>
      <h1>{stage.label}</h1>
      <p>
        {stage.detail}。这一阶段不能靠用户点击直接越过；必须先完成并冻结 S0
        问题契约，随后由 Graph 开放下一前沿。
      </p>
      <div className="locked-responsibility">
        <span>本阶段主要责任</span>
        <strong>{stage.owner}</strong>
      </div>
      <button
        className="secondary-button"
        type="button"
        onClick={() => setSelectedStage("S0")}
      >
        返回 S0 定义问题
      </button>
    </section>
  );
}

function ContextRail({ bridge }: { bridge: StudioBridge }) {
  return (
    <aside className="context-rail">
      <section className="rail-card bridge-card">
        <div className="rail-heading">
          <span>本地执行桥</span>
          <small>{bridge.connected ? "CONNECTED" : "OFFLINE"}</small>
        </div>
        <label>
          <span>Bridge URL</span>
          <input
            value={bridge.url}
            onChange={(event) => bridge.setUrl(event.target.value)}
            inputMode="url"
            spellCheck={false}
          />
        </label>
        <label>
          <span>一次性会话令牌</span>
          <input
            value={bridge.token}
            onChange={(event) => bridge.setToken(event.target.value)}
            type="password"
            autoComplete="off"
            placeholder="只保存在当前页面内存"
          />
        </label>
        <button
          type="button"
          onClick={() => void bridge.connect()}
          disabled={bridge.busy}
        >
          {bridge.connected ? "重新检查连接" : "连接本地 FMA"}
        </button>
        <p>
          authority key 留在本地服务端，永不进入浏览器、模型提示或前端日志。
        </p>
      </section>

      <section className="rail-card">
        <div className="rail-heading">
          <span>本次建模怎样完成</span>
          <small>Graph 贯穿其中</small>
        </div>
        <div className="loop-list">
          {modelingLoop.map((step, index) => (
            <div
              className={`loop-item loop-${step.status}`}
              key={step.id}
            >
              <span>{String(index + 1).padStart(2, "0")}</span>
              <div>
                <strong>{step.label}</strong>
                <p>{step.detail}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="rail-card">
        <div className="rail-heading">
          <span>三种责任不混合</span>
          <small>不是让人替 Agent 建模</small>
        </div>
        <div className="role-list">
          {authorityRoles.map((item) => (
            <article key={item.role}>
              <span className={`role-dot role-${item.tone}`} />
              <div>
                <strong>{item.role}</strong>
                <small>{item.action}</small>
                <p>{item.description}</p>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="rail-card rail-boundary">
        <span>当前产品状态</span>
        <strong>
          {bridge.connected ? "本地执行桥已连接" : "前端工作台已就绪"}
        </strong>
        <p>
          {bridge.connected
            ? "网页现在可以创建真实 FMA 工作区并启动受控 S0；S1–S6 仍按阶段逐步接入。"
            : "在本机启动 Studio Bridge 后，网页可以创建真实任务并让 Codex 完成受控 S0。"}
        </p>
      </section>
    </aside>
  );
}

function GraphView() {
  const [selectedNode, setSelectedNode] = useState("recovery");
  const node = graphNodes.find((item) => item.id === selectedNode)!;

  return (
    <div className="artifact-view">
      <header className="artifact-header">
        <div>
          <span className="section-kicker plain">REAL RUN · ITERATION 36</span>
          <h1>一次真实建模，不是一次猜中。</h1>
          <p>
            这里保留了初始失败、换表征、候选淘汰与最终冻结预测。它是工作台未来执行新任务时应持续产生的可审计过程。
          </p>
        </div>
        <Pill tone="green">{completedTask.status}</Pill>
      </header>

      <section className="work-card run-summary">
        <div>
          <span>任务</span>
          <strong>{completedTask.title}</strong>
        </div>
        <div>
          <span>最终候选</span>
          <strong>{completedTask.selectedModel}</strong>
        </div>
        <div>
          <span>私有评估</span>
          <strong className="amber-text">NOT RUN · 0/1</strong>
        </div>
      </section>

      <section className="work-card graph-card">
        <div className="card-heading">
          <div>
            <span className="section-kicker plain">MODEL LINEAGE</span>
            <h2>失败与恢复都留在同一张图里</h2>
          </div>
          <span className="required-note">选择节点查看证据</span>
        </div>
        <div className="run-graph" aria-label="I36 候选演化图">
          {graphNodes.map((item, index) => (
            <div className="run-node-wrap" key={item.id}>
              <button
                className={`run-node run-${item.tone} ${
                  item.id === selectedNode ? "selected" : ""
                }`}
                type="button"
                onClick={() => setSelectedNode(item.id)}
                aria-pressed={item.id === selectedNode}
              >
                <small>{item.status}</small>
                <strong>{item.label}</strong>
              </button>
              {index < graphNodes.length - 1 && <span>→</span>}
            </div>
          ))}
        </div>
        <article className={`node-detail detail-${node.tone}`}>
          <div>
            <span>SELECTED NODE · {node.status}</span>
            <h3>{node.title}</h3>
            <p>{node.description}</p>
          </div>
          <dl>
            {node.facts.map((fact) => (
              <div key={fact.label}>
                <dt>{fact.label}</dt>
                <dd>{fact.value}</dd>
              </div>
            ))}
          </dl>
        </article>
      </section>
    </div>
  );
}

function DeliveryView() {
  return (
    <div className="artifact-view">
      <header className="artifact-header">
        <div>
          <span className="section-kicker plain">EVIDENCE & DELIVERY</span>
          <h1>交付的不只是答案，<br />而是一条可复查的论证链。</h1>
          <p>
            用户得到模型、代码、预测与报告；Verifier
            得到精确输入、检查收据和失败历史；最终决策仍由价值所有者负责。
          </p>
        </div>
        <Pill tone="amber">外部资格未授予</Pill>
      </header>

      <section className="delivery-grid">
        <div className="work-card evidence-card">
          <div className="card-heading">
            <div>
              <span className="section-kicker plain">L0–L4</span>
              <h2>I36 公开证据</h2>
            </div>
          </div>
          <div className="evidence-list">
            {evidenceLevels.map((item) => (
              <article key={item.level}>
                <span>{item.level}</span>
                <div>
                  <strong>{item.title}</strong>
                  <p>{item.evidence}</p>
                </div>
                <Pill tone="green">{item.status}</Pill>
              </article>
            ))}
          </div>
        </div>

        <div className="delivery-stack">
          <section className="work-card prediction-card">
            <div className="card-heading">
              <div>
                <span className="section-kicker plain">FROZEN OUTPUT</span>
                <h2>四步注册预测</h2>
              </div>
            </div>
            <div className="prediction-values">
              {completedTask.predictions.map((value, index) => (
                <div key={value}>
                  <span>H{index + 1}</span>
                  <strong>{value.toFixed(2)}</strong>
                </div>
              ))}
            </div>
          </section>

          <section className="authority-warning">
            <span>AUTHORITY BOUNDARY</span>
            <h2>能完成公开建模，不等于能给自己授予科学资格。</h2>
            <p>
              外部私测仍为 NOT RUN；科学资格与现实行动权限均为 false。前端不能把流程进度渲染成更强的结论。
            </p>
          </section>
        </div>
      </section>
    </div>
  );
}

export default function Home() {
  const [view, setView] = useState<ViewId>("workspace");
  const [selectedStage, setSelectedStage] = useState<string>("S0");
  const [objective, setObjective] = useState("");
  const bridge = useStudioBridge();
  const viewingCompletedTask = view !== "workspace";

  return (
    <div className="studio-shell">
      <header className="studio-topbar">
        <button
          className="brand"
          type="button"
          onClick={() => setView("workspace")}
          aria-label="返回 FMA 建模工作台"
        >
          <span className="brand-mark">F</span>
          <span>
            <strong>FMA</strong>
            <small>MODELING STUDIO · V5.7</small>
          </span>
        </button>

        <nav className="view-tabs" aria-label="主视图">
          <button
            type="button"
            className={view === "workspace" ? "active" : ""}
            onClick={() => setView("workspace")}
          >
            建模工作台
          </button>
          <button
            type="button"
            className={view === "graph" ? "active" : ""}
            onClick={() => setView("graph")}
          >
            运行图
          </button>
          <button
            type="button"
            className={view === "delivery" ? "active" : ""}
            onClick={() => setView("delivery")}
          >
            证据与交付
          </button>
        </nav>

        <div className="runtime-state">
          <span className="runtime-dot" />
          <span>
            {bridge.connected ? "本地内核已连接" : "前端已就绪"}
            <small>
              {bridge.connected ? "S0 真实执行可用" : "执行服务待连接"}
            </small>
          </span>
        </div>
      </header>

      <aside className="studio-sidebar">
        <button
          className="new-task-button"
          type="button"
          onClick={() => {
            setView("workspace");
            setSelectedStage("S0");
          }}
        >
          <span>+</span>
          新建真实任务
        </button>

        <div className="task-switcher">
          <p>任务</p>
          <button
            className={!viewingCompletedTask ? "selected" : ""}
            type="button"
            onClick={() => setView("workspace")}
          >
            <span className="task-status task-draft" />
            <span>
              <strong>未命名建模任务</strong>
              <small>本地草案 · S0</small>
            </span>
          </button>
          <button
            className={viewingCompletedTask ? "selected" : ""}
            type="button"
            onClick={() => setView("graph")}
          >
            <span className="task-status task-complete" />
            <span>
              <strong>{completedTask.shortId}</strong>
              <small>公开运行 · 已完成</small>
            </span>
          </button>
        </div>

        <div className="sidebar-divider" />
        <p className="sidebar-label">建模阶段</p>
        <StageNavigator
          selectedStage={selectedStage}
          setSelectedStage={(stage) => {
            setSelectedStage(stage);
            if (viewingCompletedTask) setView("workspace");
          }}
          completed={false}
        />

        <div className="sidebar-foot">
          <span>控制面</span>
          <strong>Graph-native S0–S6</strong>
          <small>阶段不能由模型自审或跳过</small>
        </div>
      </aside>

      <main className="studio-main">
        {view === "workspace" && (
          <div className="workspace-layout">
            <div className="work-column">
              {selectedStage === "S0" ? (
                <IntakeWorkspace
                  objective={objective}
                  setObjective={setObjective}
                  bridge={bridge}
                />
              ) : (
                <LockedStage
                  stageId={selectedStage}
                  setSelectedStage={setSelectedStage}
                />
              )}
            </div>
            <ContextRail bridge={bridge} />
          </div>
        )}
        {view === "graph" && <GraphView />}
        {view === "delivery" && <DeliveryView />}
      </main>
    </div>
  );
}
