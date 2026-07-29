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
import {
  type DecisionUseRequestV62,
  type EvidenceStatus,
  type EvidenceScope,
  type StudioWorldBankDataRequestV62,
  type WorkflowMode,
  useStudioBridge,
} from "./use-studio-bridge";

type ViewId = "workspace" | "graph" | "delivery";
type StudioBridge = ReturnType<typeof useStudioBridge>;
type PillTone = "neutral" | "blue" | "green" | "amber" | "red";

const queryTypes = ["自动判断", "解释", "预测", "优化", "控制", "设计"];
const compactInputStyle: React.CSSProperties = {
  background: "#0d1015",
  border: "1px solid rgba(255, 255, 255, 0.14)",
  borderRadius: 8,
  color: "#f2f4f8",
  fontSize: 12,
  minWidth: 0,
  padding: "10px 11px",
  width: "100%",
};
const compactFormGridStyle: React.CSSProperties = {
  display: "grid",
  gap: 10,
  gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
  marginTop: 14,
};

function Pill({
  children,
  tone = "neutral",
}: {
  children: React.ReactNode;
  tone?: PillTone;
}) {
  return <span className={`pill pill-${tone}`}>{children}</span>;
}

function evidenceTone(status: EvidenceStatus): PillTone {
  if (status === "PASS") return "green";
  if (status === "FAIL") return "red";
  return "amber";
}

function StageNavigator({
  selectedStage,
  setSelectedStage,
  completed = false,
  stageStatuses,
}: {
  selectedStage: string;
  setSelectedStage: (stage: string) => void;
  completed?: boolean;
  stageStatuses?: Record<string, string>;
}) {
  return (
    <div className="stage-list" aria-label="S0 到 S6 建模阶段">
      {stages.map((stage, index) => {
        const isActive = selectedStage === stage.id;
        const stageStatus = stageStatuses?.[stage.id];
        const isLocked =
          !completed &&
          stage.id !== "S0" &&
          !["frontier", "gate_open"].includes(stageStatus ?? "");
        const gateOpen = stageStatus === "gate_open";
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
              {completed || gateOpen ? "✓" : index === 0 ? "●" : index + 1}
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

function S1Workspace({ bridge }: { bridge: StudioBridge }) {
  const s1Open = bridge.task?.workflow.stage_statuses.S1 === "gate_open";
  const running = bridge.task
    ? ["accepted", "running"].includes(bridge.task.activity)
    : false;
  const epistemic = bridge.task?.epistemic;
  const predataPrepared = bridge.task?.predata_v67.prepared === true;
  const predataRecoveryPending =
    bridge.task?.predata_v67.transaction_status === "RECOVERY_PENDING";
  const predataStalePending =
    bridge.task?.predata_v67.transaction_status === "STALE_PENDING";
  const predataRecoveryAvailable =
    bridge.task?.predata_v67.recovery_available === true;
  const predataRequestFrozen =
    predataPrepared || predataRecoveryPending || predataStalePending;
  const usePredataV67 =
    bridge.task?.predata_v67.workflow_mode === "v67";
  const [predataJson, setPredataJson] = useState(() =>
    JSON.stringify(
      bridge.preparedPredataRequest ??
        ({
          schema_version: "6.2",
          adapter_id: "scalar_autonomous_ode_v52",
          contract_id: "",
          country_code: "",
          indicator_id: "",
          start_year: 2000,
          end_year: 2024,
          minimum_observations: 23,
          state_unit: "",
          attribution: "World Bank Open Data API",
          semantic_name: "",
          operational_definition: "",
          observation_time_basis: "calendar year",
          aggregation_level: "country",
          fixture_only: bridge.taskEvidenceScope === "development",
        } satisfies StudioWorldBankDataRequestV62),
      null,
      2,
    ),
  );
  const [predataError, setPredataError] = useState("");
  const branches = [
    ["MECHANISTIC", "机制与状态演化"],
    ["NULL BASELINE", "最小可击败基线"],
    ["STATISTICAL", "统计结构与不确定性"],
    ["SYSTEM LEARNING", "系统辨识与函数学习"],
  ];

  const prepareOrReconcilePredata = async () => {
    setPredataError("");
    try {
      if (predataStalePending) {
        throw new Error(
          "This pre-data intent is bound to a stale S0 gate. Recover the modeling graph before preparing a new transaction.",
        );
      }
      if (predataRecoveryPending) {
        await bridge.reconcilePredata();
        return;
      }
      const parsed = JSON.parse(predataJson) as StudioWorldBankDataRequestV62;
      await bridge.preparePredata(parsed);
    } catch (reason) {
      setPredataError(
        reason instanceof Error
          ? reason.message
          : "V6.7 pre-data preparation failed",
      );
    }
  };

  return (
    <div className="s1-workspace">
      <section className="work-hero">
        <div className="work-hero-copy">
          <div className="section-kicker">
            <span>ACTIVE STAGE · S1</span>
            <Pill tone={s1Open ? "green" : "blue"}>
              {s1Open ? "候选已冻结" : "并行探索可启动"}
            </Pill>
          </div>
          <h1>先隔离地产生差异，<br />再受控地共享知识。</h1>
          <p>
            四个新鲜 Codex 角色先在看不到同伴结论的条件下提出结构不同的候选。
            Harness 冻结来源后，Broker 才开放有限共享，并把迁移结论保留为待检验假设。
          </p>
        </div>
        <button
          className="primary-button"
          type="button"
          disabled={
            running ||
            bridge.busy ||
            Boolean(s1Open) ||
            (usePredataV67 && !predataPrepared)
          }
          onClick={() => void bridge.runS1()}
        >
          {s1Open
            ? "S1 GATE 已打开"
            : running
              ? "并行角色正在运行…"
              : "启动 Graph-native S1"}
          <span aria-hidden="true">→</span>
        </button>
      </section>

      <section className="work-card">
        <div className="card-heading">
          <div>
            <span className="section-kicker plain">
              S0.5 · PRE-DATA CONTRACT
            </span>
            <h2>Choose the compatible S1 entry path</h2>
          </div>
          <Pill
            tone={
              usePredataV67
                ? predataPrepared
                  ? "green"
                  : predataStalePending
                    ? "red"
                    : "amber"
                : "neutral"
            }
          >
            {usePredataV67
              ? predataPrepared
                ? "V6.7 PREPARED"
                : predataRecoveryPending
                  ? "V6.7 RECOVERY PENDING"
                : predataStalePending
                  ? "V6.7 STALE S0 INTENT"
                : "PREPARATION REQUIRED"
              : "LEGACY COMPATIBLE"}
          </Pill>
        </div>
        <div className="segmented" role="group" aria-label="S1 entry path">
          <button
            type="button"
            className={!usePredataV67 ? "selected" : ""}
            aria-pressed={!usePredataV67}
            disabled
          >
            Legacy S1
          </button>
          <button
            type="button"
            className={usePredataV67 ? "selected" : ""}
            aria-pressed={usePredataV67}
            disabled
          >
            V6.7 pre-data first
          </button>
        </div>
        <p className="s1-not-run">
          The entry path is authority-frozen when the task is created and
          cannot be changed after S0 begins.
        </p>
        {usePredataV67 ? (
          <>
            <label className="objective-field" style={{ marginTop: 14 }}>
              <span>Typed World Bank pre-data request</span>
              <textarea
                value={predataJson}
                onChange={(event) => {
                  setPredataJson(event.target.value);
                  setPredataError("");
                }}
                disabled={predataRequestFrozen || Boolean(s1Open) || running}
                rows={16}
                spellCheck={false}
              />
              <small>
                This freezes source semantics, measurement design, and the exact
                execution protocol before S1 or observation access.
              </small>
            </label>
            <div className="backhalf-actions">
              <button
                className="secondary-button"
                type="button"
                disabled={
                  bridge.busy ||
                  running ||
                  predataPrepared ||
                  predataStalePending ||
                  (predataRecoveryPending && !predataRecoveryAvailable) ||
                  !predataJson.trim()
                }
                onClick={() => void prepareOrReconcilePredata()}
              >
                {predataPrepared
                  ? "V6.7 pre-data contracts frozen"
                  : predataRecoveryPending
                    ? "Resume exact pre-data transaction"
                  : predataStalePending
                    ? "Stale pre-data intent requires graph recovery"
                  : "Prepare V6.7 pre-data contracts"}
              </button>
            </div>
            {predataStalePending && (
              <p className="bridge-error">
                This intent is bound to an obsolete S0 gate. Inspect and recover
                the modeling graph before starting a new pre-data transaction;
                the frozen request remains visible for audit.
              </p>
            )}
          </>
        ) : (
          <p className="s1-not-run">
            Legacy tasks may continue directly to S1 and freeze their data
            afterward. Selecting the V6.7 path makes pre-data preparation
            mandatory before S1.
          </p>
        )}
        {(predataError || bridge.error) && (
          <p className="bridge-error">{predataError || bridge.error}</p>
        )}
      </section>

      <section className="work-card s1-branch-card">
        <div className="card-heading">
          <div>
            <span className="section-kicker plain">BLIND FRONTIER</span>
            <h2>四条独立候选链路</h2>
          </div>
          <span className="required-note">共享 S0 · 隔离来源</span>
        </div>
        <div className="s1-branch-grid">
          {branches.map(([id, label], index) => (
            <article key={id}>
              <small>BRANCH {String(index + 1).padStart(2, "0")}</small>
              <strong>{id}</strong>
              <p>{label}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="work-card s1-knowledge-card">
        <div className="card-heading">
          <div>
            <span className="section-kicker plain">EPISTEMIC GRAPH · V5.8</span>
            <h2>共享知识不等于共享结论</h2>
          </div>
          <Pill tone={epistemic?.independence_passed ? "green" : "amber"}>
            {epistemic
              ? epistemic.independence_passed
                ? "ORIGIN ISOLATION PASS"
                : "ORIGIN ISOLATION FAIL"
              : "NOT RUN"}
          </Pill>
        </div>
        <div className="knowledge-flow">
          <span>盲候选</span>
          <i>→</i>
          <span>来源冻结</span>
          <i>→</i>
          <span>Broker 披露</span>
          <i>→</i>
          <span>跨范式翻译</span>
          <i>→</i>
          <span>目标分支复核</span>
        </div>
        {epistemic ? (
          <div className="epistemic-summary expanded">
            <div>
              <span>知识单元</span>
              <strong>{epistemic.knowledge_unit_count}</strong>
            </div>
            <div>
              <span>隔离来源分支</span>
              <strong>{epistemic.effective_independent_branches}</strong>
            </div>
            <div>
              <span>披露包</span>
              <strong>{epistemic.disclosure_packet_count}</strong>
            </div>
            <div>
              <span>迁移评估</span>
              <strong>{epistemic.transfer_assessment_count}</strong>
            </div>
          </div>
        ) : (
          <p className="s1-not-run">
            尚未运行。来源隔离不等于独立科学复现；跨任务经验默认隔离，
            本阶段不能自行授予科学支持。
          </p>
        )}
      </section>
    </div>
  );
}

function BackhalfWorkspace({
  bridge,
  stageId,
}: {
  bridge: StudioBridge;
  stageId: string;
}) {
  const preparedPredataRequest = bridge.preparedPredataRequest;
  const predataPrepared = bridge.task?.predata_v67.prepared === true;
  const [dataMode, setDataMode] = useState<"raw_ode" | "world_bank">(
    predataPrepared ? "world_bank" : "raw_ode",
  );
  const [dataJson, setDataJson] = useState("");
  const [worldBankDraft, setWorldBankDraft] = useState(() => ({
    adapter_id:
      preparedPredataRequest?.adapter_id ??
      ("scalar_autonomous_ode_v52" as
        | "scalar_autonomous_ode_v52"
        | "adaptive_positive_series_v57"),
    contract_id: preparedPredataRequest?.contract_id ?? "",
    country_code: preparedPredataRequest?.country_code ?? "",
    indicator_id: preparedPredataRequest?.indicator_id ?? "",
    start_year: preparedPredataRequest
      ? String(preparedPredataRequest.start_year)
      : "",
    end_year: preparedPredataRequest
      ? String(preparedPredataRequest.end_year)
      : "",
    minimum_observations: preparedPredataRequest
      ? String(preparedPredataRequest.minimum_observations)
      : "23",
    state_unit: preparedPredataRequest?.state_unit ?? "",
    attribution: preparedPredataRequest?.attribution ?? "",
    semantic_name: preparedPredataRequest?.semantic_name ?? "",
    operational_definition:
      preparedPredataRequest?.operational_definition ?? "",
    observation_time_basis:
      preparedPredataRequest?.observation_time_basis ?? "calendar year",
    aggregation_level:
      preparedPredataRequest?.aggregation_level ?? "country",
    fixture_only: preparedPredataRequest?.fixture_only ?? false,
  }));
  const [localError, setLocalError] = useState("");
  const backhalf = bridge.task?.backhalf;
  const recovery = bridge.task?.recovery;
  const scientificSuccess = bridge.task?.scientific_success;
  const scientificClosure = bridge.task?.scientific_closure;
  const running = bridge.task
    ? ["accepted", "running"].includes(bridge.task.activity)
    : false;
  const workflowComplete = backhalf?.workflow_complete ?? false;
  const worldBankRequestLocked =
    Boolean(backhalf?.data_received) || preparedPredataRequest !== null;
  const closureDimensionStatus = (dimensionId: string) => {
    const dimensions = scientificClosure?.dimensions;
    if (!dimensions) return undefined;
    if (Array.isArray(dimensions)) {
      return dimensions.find((item) => item.dimension_id === dimensionId)
        ?.status;
    }
    return dimensions[dimensionId]?.status;
  };
  const localLevelStatuses = ["L0", "L1", "L2", "L3", "L4"].map(
    (level) => backhalf?.level_statuses[level],
  );
  const localL0L4Status: EvidenceStatus =
    localLevelStatuses.every((status) => status === "PASS") &&
    backhalf?.scientific_acceptance
      ? "PASS"
      : localLevelStatuses.some((status) => status === "FAIL") ||
          (localLevelStatuses.every(Boolean) &&
            !backhalf?.scientific_acceptance)
        ? "FAIL"
        : "NOT_RUN";
  const closureEvaluated = scientificClosure?.evaluated === true;
  const sourceIntegrityStatus =
    (closureEvaluated
      ? scientificClosure?.source_integrity_status
      : undefined) ??
    backhalf?.source_integrity_status ??
    "NOT_RUN";
  const scientificProvenanceStatus =
    (closureEvaluated
      ? scientificClosure?.scientific_provenance_status
      : undefined) ??
    backhalf?.scientific_provenance_status ??
    closureDimensionStatus("data_provenance") ??
    "NOT_RUN";
  const decisionEvidenceStatus =
    (closureEvaluated
      ? scientificClosure?.decision_evidence_status
      : undefined) ??
    backhalf?.decision_evidence_status ??
    "NOT_RUN";
  const scientificDecisionStatus =
    (closureEvaluated
      ? scientificClosure?.scientific_decision_status
      : undefined) ??
    backhalf?.scientific_decision_status ??
    closureDimensionStatus("decision_value") ??
    "NOT_RUN";
  const stageAdmissionStatus =
    (closureEvaluated
      ? scientificClosure?.stage_admission_status
      : undefined) ?? "NOT_RUN";
  const closureVerificationStatus =
    (closureEvaluated
      ? scientificClosure?.closure_verification_status
      : undefined) ?? "NOT_RUN";
  const localEvidenceStatus =
    (closureEvaluated
      ? scientificClosure?.local_evidence_status
      : undefined) ?? "NOT_RUN";
  const scientificClosureStatus =
    (closureEvaluated
      ? scientificClosure?.scientific_closure_status
      : undefined) ?? "NOT_RUN";
  const rollingConfirmationStatus =
    scientificSuccess?.confirmation?.status ??
    backhalf?.rolling_confirmation_status ??
    "NOT_RUN";

  const freezeData = async () => {
    setLocalError("");
    try {
      const parsed = JSON.parse(dataJson) as Record<string, unknown>;
      await bridge.ingestOdeData(parsed);
    } catch (reason) {
      setLocalError(
        reason instanceof Error ? reason.message : "无法解析或冻结 ODE 数据",
      );
    }
  };

  const updateWorldBankDraft = (
    field: keyof typeof worldBankDraft,
    value: string | boolean,
  ) => {
    setWorldBankDraft((current) => ({ ...current, [field]: value }));
    setLocalError("");
  };

  const freezeWorldBankData = async () => {
    setLocalError("");
    try {
      const requiredText = [
        ["contract ID", worldBankDraft.contract_id],
        ["country code", worldBankDraft.country_code],
        ["indicator ID", worldBankDraft.indicator_id],
        ["state unit", worldBankDraft.state_unit],
        ["attribution", worldBankDraft.attribution],
        ["semantic name", worldBankDraft.semantic_name],
        ["operational definition", worldBankDraft.operational_definition],
        ["observation time basis", worldBankDraft.observation_time_basis],
        ["aggregation level", worldBankDraft.aggregation_level],
      ] as const;
      const missing = requiredText
        .filter(([, value]) => !value.trim())
        .map(([label]) => label);
      if (missing.length) {
        throw new Error(`World Bank 来源字段未填写完整：${missing.join("、")}`);
      }
      const startYear = Number(worldBankDraft.start_year);
      const endYear = Number(worldBankDraft.end_year);
      const minimumObservations = Number(
        worldBankDraft.minimum_observations,
      );
      if (
        !worldBankDraft.start_year.trim() ||
        !worldBankDraft.end_year.trim() ||
        !Number.isInteger(startYear) ||
        !Number.isInteger(endYear) ||
        endYear < startYear
      ) {
        throw new Error("起止年份必须是整数，且结束年份不能早于开始年份。");
      }
      if (
        !worldBankDraft.minimum_observations.trim() ||
        !Number.isInteger(minimumObservations) ||
        minimumObservations < 1
      ) {
        throw new Error("最少观测数必须是大于 0 的整数。");
      }
      if (
        worldBankDraft.attribution.trim().length < 10 ||
        worldBankDraft.operational_definition.trim().length < 10
      ) {
        throw new Error("来源署名和操作定义都至少需要 10 个字符。");
      }
      if (
        worldBankDraft.fixture_only &&
        bridge.taskEvidenceScope !== "development"
      ) {
        throw new Error("fixture_only 仅允许用于 development 任务。");
      }
      if (
        !worldBankDraft.fixture_only &&
        bridge.taskEvidenceScope !== "public_data"
      ) {
        throw new Error(
          "真实 World Bank 获取必须在创建任务时选择 public_data evidence scope。",
        );
      }
      const payload: StudioWorldBankDataRequestV62 = {
        schema_version: "6.2",
        adapter_id: worldBankDraft.adapter_id,
        contract_id: worldBankDraft.contract_id.trim(),
        country_code: worldBankDraft.country_code.trim(),
        indicator_id: worldBankDraft.indicator_id.trim(),
        start_year: startYear,
        end_year: endYear,
        minimum_observations: minimumObservations,
        state_unit: worldBankDraft.state_unit.trim(),
        attribution: worldBankDraft.attribution.trim(),
        semantic_name: worldBankDraft.semantic_name.trim(),
        operational_definition:
          worldBankDraft.operational_definition.trim(),
        observation_time_basis:
          worldBankDraft.observation_time_basis.trim(),
        aggregation_level: worldBankDraft.aggregation_level.trim(),
        fixture_only: worldBankDraft.fixture_only,
      };
      await bridge.ingestWorldBankData(payload);
    } catch (reason) {
      setLocalError(
        reason instanceof Error
          ? reason.message
          : "无法登记 World Bank 官方数据源",
      );
    }
  };

  return (
    <div className="backhalf-workspace">
      <section className="work-hero">
        <div className="work-hero-copy">
          <div className="section-kicker">
            <span>EXECUTABLE BACK HALF · {stageId}</span>
            <Pill tone={workflowComplete ? "green" : "blue"}>
              {workflowComplete ? "S2–S6 COMPLETE" : "NARROW ODE PATH"}
            </Pill>
          </div>
          <h1>先把证据链跑到底，<br />再提升候选质量。</h1>
          <p>
            当前生产级适配器只接受正值标量时间序列，并在固定的 constant、
            exponential、gompertz、logistic 候选族上执行 L0–L4。
            S1 若没有明确选择兼容的自治 ODE 形式，链路会拒绝启动。
          </p>
        </div>
      </section>

      <section className="work-card backhalf-intake-card">
        <div className="card-heading">
          <div>
            <span className="section-kicker plain">S2 · DATA FREEZE</span>
            <h2>冻结正值标量序列与来源</h2>
          </div>
          <Pill tone={backhalf?.data_received ? "green" : "amber"}>
            {backhalf?.data_received ? "DATA RECEIVED" : "INPUT REQUIRED"}
          </Pill>
        </div>
        <div className="control-block" style={{ marginBottom: 16 }}>
          <span className="control-label">互斥数据入口</span>
          <div className="segmented" role="group" aria-label="选择数据入口">
            <button
              type="button"
              className={dataMode === "raw_ode" ? "selected" : ""}
              aria-pressed={dataMode === "raw_ode"}
              disabled={Boolean(backhalf?.data_received) || predataPrepared}
              onClick={() => {
                setDataMode("raw_ode");
                setLocalError("");
              }}
            >
              原始 ODE JSON
            </button>
            <button
              type="button"
              className={dataMode === "world_bank" ? "selected" : ""}
              aria-pressed={dataMode === "world_bank"}
              disabled={backhalf?.data_received}
              onClick={() => {
                setDataMode("world_bank");
                setLocalError("");
              }}
            >
              World Bank 注册源
            </button>
          </div>
        </div>
        {dataMode === "raw_ode" ? (
          <label className="objective-field">
            <span>正值标量序列 JSON</span>
            <textarea
              value={dataJson}
              onChange={(event) => {
                setDataJson(event.target.value);
                setLocalError("");
              }}
              disabled={backhalf?.data_received}
              placeholder={`{
  "time_unit": "day",
  "state_unit": "count",
  "times": [0, 1, 2, "...至少 12 个严格递增时间点"],
  "observations": [5.1, 6.2, 7.4, "...对应的正值观测"],
  "source_id": "你的数据来源",
  "license_status": "user-provided",
  "fixture_only": false
}`}
              rows={12}
              spellCheck={false}
            />
            <small>
              Harness 会先写入原始文件并记录哈希；用户提供的来源声明不是第三方来源认证。
            </small>
          </label>
        ) : (
          <div className="control-block">
            <span className="control-label">
              World Bank 官方 API 注册源 · task scope:{" "}
              {bridge.taskEvidenceScope ?? "unknown"}
            </span>
            {predataPrepared && (
              <small
                style={{
                  color: "#9aa5b5",
                  display: "block",
                  marginTop: 8,
                }}
              >
                V6.7 pre-data contracts were frozen before S1. The official
                source intake must replay the exact registered request.
              </small>
            )}
            <div style={compactFormGridStyle}>
              <label>
                <span className="control-label">适配器</span>
                <select
                  style={compactInputStyle}
                  value={worldBankDraft.adapter_id}
                  disabled={worldBankRequestLocked}
                  onChange={(event) =>
                    updateWorldBankDraft("adapter_id", event.target.value)
                  }
                >
                  <option value="scalar_autonomous_ode_v52">
                    scalar autonomous ODE
                  </option>
                  <option value="adaptive_positive_series_v57">
                    adaptive positive series
                  </option>
                </select>
              </label>
              {[
                ["contract_id", "来源契约 ID", "wb-population-cn"],
                ["country_code", "国家代码", "CHN"],
                ["indicator_id", "指标 ID", "SP.POP.TOTL"],
                ["state_unit", "状态单位", "persons"],
                ["semantic_name", "语义名称", "resident population"],
                ["attribution", "来源署名", "World Bank Open Data API"],
                [
                  "observation_time_basis",
                  "观测时间基准",
                  "calendar year",
                ],
                ["aggregation_level", "聚合层级", "country"],
              ].map(([field, label, placeholder]) => (
                <label key={field}>
                  <span className="control-label">{label}</span>
                  <input
                    style={compactInputStyle}
                    value={String(
                      worldBankDraft[
                        field as keyof typeof worldBankDraft
                      ],
                    )}
                    disabled={worldBankRequestLocked}
                    placeholder={placeholder}
                    spellCheck={false}
                    onChange={(event) =>
                      updateWorldBankDraft(
                        field as keyof typeof worldBankDraft,
                        event.target.value,
                      )
                    }
                  />
                </label>
              ))}
              {[
                ["start_year", "开始年份", "2000"],
                ["end_year", "结束年份", "2024"],
                ["minimum_observations", "最少观测数", "23"],
              ].map(([field, label, placeholder]) => (
                <label key={field}>
                  <span className="control-label">{label}</span>
                  <input
                    style={compactInputStyle}
                    type="number"
                    value={String(
                      worldBankDraft[
                        field as keyof typeof worldBankDraft
                      ],
                    )}
                    disabled={worldBankRequestLocked}
                    placeholder={placeholder}
                    onChange={(event) =>
                      updateWorldBankDraft(
                        field as keyof typeof worldBankDraft,
                        event.target.value,
                      )
                    }
                  />
                </label>
              ))}
            </div>
            <label className="objective-field" style={{ marginTop: 12 }}>
              <span>操作定义</span>
              <textarea
                value={worldBankDraft.operational_definition}
                disabled={worldBankRequestLocked}
                placeholder="明确这个指标在本任务中测量什么、如何聚合，以及不能解释为什么。"
                rows={3}
                onChange={(event) =>
                  updateWorldBankDraft(
                    "operational_definition",
                    event.target.value,
                  )
                }
              />
            </label>
            <label
              style={{
                alignItems: "center",
                display: "flex",
                gap: 8,
                marginTop: 12,
              }}
            >
              <input
                type="checkbox"
                checked={worldBankDraft.fixture_only}
                disabled={
                  worldBankRequestLocked ||
                  bridge.taskEvidenceScope !== "development"
                }
                onChange={(event) =>
                  updateWorldBankDraft("fixture_only", event.target.checked)
                }
              />
              <span style={{ color: "#aab2bf", fontSize: 11 }}>
                fixture_only（默认 false；仅 development
                调试，不能建立真实能力或科学结论）
              </span>
            </label>
            <small
              style={{
                color: "#7f8998",
                display: "block",
                fontSize: 9,
                marginTop: 9,
              }}
            >
              非 fixture 获取要求任务 evidence_scope=public_data。Harness
              会抓取并认证已注册来源，机械完整性 PASS 仍不等于科学资格。
            </small>
          </div>
        )}
        <div className="backhalf-actions">
          <button
            className="secondary-button"
            type="button"
            disabled={
              bridge.busy ||
              running ||
              Boolean(backhalf?.data_received) ||
              (dataMode === "raw_ode" && !dataJson.trim())
            }
            onClick={() =>
              void (dataMode === "raw_ode"
                ? freezeData()
                : freezeWorldBankData())
            }
          >
            {dataMode === "raw_ode"
              ? "仅冻结并审计数据"
              : "获取、认证并登记来源"}
          </button>
          <button
            className="primary-button"
            type="button"
            disabled={
              bridge.busy ||
              running ||
              !backhalf?.data_received ||
              workflowComplete
            }
            onClick={() => void bridge.runBackhalf()}
          >
            {running ? "S2–S6 正在运行…" : "运行 S2–S6"}
            <span aria-hidden="true">→</span>
          </button>
        </div>
        {(localError || bridge.error) && (
          <p className="bridge-error">{localError || bridge.error}</p>
        )}
      </section>

      <section className="work-card backhalf-evidence-card">
        <div className="card-heading">
          <div>
            <span className="section-kicker plain">
              V6.1 · CLAIM-EVIDENCE PROJECTION
            </span>
            <h2>Claim-relative success, not a single green badge</h2>
          </div>
          <Pill
            tone={evidenceTone(
              scientificSuccess?.local_predictive_gate_status ?? "NOT_RUN",
            )}
          >
            LOCAL PREDICTIVE{" "}
            {scientificSuccess?.local_predictive_gate_status ?? "NOT_RUN"}
          </Pill>
        </div>
        <div className="backhalf-stage-grid">
          {[
            ["workflow_integrity", "workflow"],
            ["data_provenance", "data"],
            ["local_adapter_checks", "local L0-L4"],
            ["leakage_safe_confirmation", "confirmation"],
            ["decision_value", "decision"],
            ["mechanism_identification", "mechanism"],
            ["external_generalization", "generalization"],
            ["scientific_qualification", "qualification"],
          ].map(([dimension, label]) => (
            <article key={dimension}>
              <span>{label}</span>
              <strong>
                {scientificSuccess?.dimensions[dimension]?.status ?? "NOT_RUN"}
              </strong>
            </article>
          ))}
        </div>
        <div className="truth-note">
          <span aria-hidden="true">i</span>
          local predictive gate:{" "}
          {scientificSuccess?.local_predictive_gate_status ?? "NOT_RUN"} ·
          claim ceiling:{" "}
          {scientificSuccess?.claim_ceiling ?? "no_scientific_claim"} ·
          rolling confirmation:{" "}
          {scientificSuccess?.confirmation
            ? `${scientificSuccess.confirmation.completed_fold_count}/${scientificSuccess.confirmation.requested_fold_count} · ${rollingConfirmationStatus}`
            : rollingConfirmationStatus}{" "}
          · V6.1 claim-evidence aggregate (not qualification):{" "}
          {scientificSuccess?.scientific_success_status ?? "NOT_RUN"} ·
          qualification: false · action authority: false
        </div>
      </section>

      <section className="work-card backhalf-evidence-card">
        <div className="card-heading">
          <div>
            <span className="section-kicker plain">
              V6.2 · SCIENTIFIC CLOSURE
            </span>
            <h2>机械完整性、局部证据与科学资格分别陈述</h2>
          </div>
          <Pill tone={evidenceTone(scientificClosureStatus)}>
            SCIENTIFIC CLOSURE {scientificClosureStatus}
          </Pill>
        </div>
        <div className="backhalf-stage-grid">
          {[
            ["source integrity (mechanical)", sourceIntegrityStatus],
            ["scientific provenance", scientificProvenanceStatus],
            [
              "source manifest admission",
              backhalf?.source_stage_admission_status ?? "NOT_RUN",
            ],
            [
              "typed semantics admission",
              backhalf?.executable_candidate_admission_status ?? "NOT_RUN",
            ],
            [
              "executed semantics",
              backhalf?.executable_candidate_status ?? "NOT_RUN",
            ],
            ["rolling confirmation", rollingConfirmationStatus],
            [
              "rolling manifest admission",
              backhalf?.rolling_confirmation_admission_status ?? "NOT_RUN",
            ],
            ["decision replay (mechanical)", decisionEvidenceStatus],
            [
              "decision manifest admission",
              backhalf?.decision_evidence_admission_status ?? "NOT_RUN",
            ],
            ["scientific decision", scientificDecisionStatus],
            ["manifest admission (mechanical)", stageAdmissionStatus],
            ["closure replay (mechanical)", closureVerificationStatus],
            ["local evidence", localEvidenceStatus],
            ["scientific closure", scientificClosureStatus],
          ].map(([label, status]) => (
            <article key={label}>
              <span>{label}</span>
              <strong>{status}</strong>
            </article>
          ))}
        </div>
        <div className="truth-note">
          <span aria-hidden="true">i</span>
          claim ceiling:{" "}
          {scientificClosure?.claim_ceiling ?? "no_scientific_claim"} ·
          fixture only:{" "}
          {String(
            scientificClosure?.fixture_only ?? backhalf?.fixture_only ?? "unknown",
          )}{" "}
          · scientific qualification: false · real-world action: false.
          Mechanical PASS means the artifact chain replayed; it is not scientific
          qualification.
        </div>
      </section>

      <section className="work-card backhalf-evidence-card">
        <div className="card-heading">
          <div>
            <span className="section-kicker plain">LIVE EVIDENCE</span>
            <h2>阶段、科学检查与权限分开显示</h2>
          </div>
          <Pill tone={evidenceTone(localL0L4Status)}>
            LOCAL L0–L4 {localL0L4Status}
          </Pill>
        </div>
        <div className="backhalf-stage-grid">
          {["S2", "S3", "S4", "S5", "S6"].map((stage) => (
            <article key={stage}>
              <span>{stage}</span>
              <strong>
                {bridge.task?.workflow.stage_statuses[stage] ?? "locked"}
              </strong>
            </article>
          ))}
        </div>
        <div className="backhalf-level-grid">
          {["L0", "L1", "L2", "L3", "L4"].map((level) => (
            <article key={level}>
              <span>{level}</span>
              <strong>{backhalf?.level_statuses[level] ?? "NOT_RUN"}</strong>
            </article>
          ))}
        </div>
        <div className="truth-note">
          <span aria-hidden="true">i</span>
          adapter: {backhalf?.adapter_id ?? "not selected"} · branch:{" "}
          {backhalf?.selected_branch ?? "not run"} · selected family:{" "}
          {backhalf?.selected_scientific_family ?? "not run"} ·
          fixture only: {String(backhalf?.fixture_only ?? "unknown")} · scientific
          qualification: false · real-world action: false · local checks do not
          establish external validity
        </div>
      </section>

      <section className="work-card backhalf-evidence-card">
        <div className="card-heading">
          <div>
            <span className="section-kicker plain">V6 · RECOVERY CONTROL</span>
            <h2>Failure becomes a bounded graph transition</h2>
          </div>
          <Pill
            tone={
              recovery?.stopped || recovery?.human_required ? "amber" : "blue"
            }
          >
            {recovery?.stopped
              ? "STOPPED"
              : recovery?.human_required
                ? "HUMAN"
                : "READY"}
          </Pill>
        </div>
        <div className="backhalf-stage-grid">
          <article>
            <span>scientific attempts</span>
            <strong>{recovery?.scientific_attempts_started ?? 1}</strong>
          </article>
          <article>
            <span>budget remaining</span>
            <strong>{recovery?.attempt_budget_remaining ?? 2}</strong>
          </article>
          <article>
            <span>same-attempt retries</span>
            <strong>{recovery?.same_attempt_retries ?? 0}</strong>
          </article>
          <article>
            <span>failure signatures</span>
            <strong>{recovery?.distinct_failure_signatures ?? 0}</strong>
          </article>
        </div>
        <div className="truth-note">
          <span aria-hidden="true">i</span>
          last action: {recovery?.last_action ?? "none"} · revoke from:{" "}
          {recovery?.last_revoke_from ?? "none"} · private adaptive feedback:
          false · stop reason:{" "}
          {recovery?.stop_reason ?? "none"} · human reason:{" "}
          {recovery?.human_reason ?? "none"} · old evidence is quarantined,
          never overwritten
        </div>
      </section>
    </div>
  );
}

function IntakeWorkspace({
  objective,
  setObjective,
  bridge,
  setSelectedStage,
}: {
  objective: string;
  setObjective: (value: string) => void;
  bridge: StudioBridge;
  setSelectedStage: (stage: string) => void;
}) {
  const [queryType, setQueryType] = useState("自动判断");
  const [files, setFiles] = useState<string[]>([]);
  const [draftCreated, setDraftCreated] = useState(false);
  const [evidenceScope, setEvidenceScope] =
    useState<EvidenceScope>("development");
  const [workflowMode, setWorkflowMode] =
    useState<WorkflowMode>("v67");
  const [decisionEnabled, setDecisionEnabled] = useState(false);
  const [decisionDraft, setDecisionDraft] = useState({
    decision_id: "",
    value_owner_ref: "",
    action_unit: "",
    underage_unit_cost: "",
    overage_unit_cost: "",
    minimum_relative_loss_improvement: "0.05",
    maximum_mean_normalized_regret: "0.20",
  });
  const [localError, setLocalError] = useState("");
  const decisionNumbers = {
    underage_unit_cost: Number(decisionDraft.underage_unit_cost),
    overage_unit_cost: Number(decisionDraft.overage_unit_cost),
    minimum_relative_loss_improvement: Number(
      decisionDraft.minimum_relative_loss_improvement,
    ),
    maximum_mean_normalized_regret: Number(
      decisionDraft.maximum_mean_normalized_regret,
    ),
  };
  const decisionComplete =
    Object.values(decisionDraft).every(
      (value) => value.trim().length > 0,
    ) &&
    decisionDraft.decision_id.trim().length > 0 &&
    decisionDraft.value_owner_ref.trim().length > 0 &&
    decisionDraft.action_unit.trim().length > 0 &&
    Number.isFinite(decisionNumbers.underage_unit_cost) &&
    decisionNumbers.underage_unit_cost > 0 &&
    Number.isFinite(decisionNumbers.overage_unit_cost) &&
    decisionNumbers.overage_unit_cost > 0 &&
    Number.isFinite(decisionNumbers.minimum_relative_loss_improvement) &&
    decisionNumbers.minimum_relative_loss_improvement >= 0 &&
    decisionNumbers.minimum_relative_loss_improvement <= 1 &&
    Number.isFinite(decisionNumbers.maximum_mean_normalized_regret) &&
    decisionNumbers.maximum_mean_normalized_regret > 0 &&
    decisionNumbers.maximum_mean_normalized_regret <= 1;
  const canCreate =
    objective.trim().length >= 12 &&
    (!decisionEnabled || decisionComplete);
  const s0Open = bridge.task?.workflow.stage_statuses.S0 === "gate_open";
  const s1Open = bridge.task?.workflow.stage_statuses.S1 === "gate_open";
  const usePredataV67 =
    bridge.task?.predata_v67.workflow_mode === "v67";
  const agentRunning = bridge.task
    ? ["accepted", "running"].includes(bridge.task.activity)
    : false;
  const epistemic = bridge.task?.epistemic;
  const portfolio = bridge.task?.portfolio_v69;

  const createTask = async () => {
    if (bridge.connected) {
      try {
        if (decisionEnabled && !decisionComplete) {
          setLocalError(
            "决策用途未填写完整：文本字段必填、两类单位代价须大于 0，阈值须位于允许范围。",
          );
          return;
        }
        const decisionUse: DecisionUseRequestV62 | undefined = decisionEnabled
          ? {
              schema_version: "6.2",
              decision_id: decisionDraft.decision_id.trim(),
              value_owner_ref: decisionDraft.value_owner_ref.trim(),
              action_unit: decisionDraft.action_unit.trim(),
              ...decisionNumbers,
            }
          : undefined;
        await bridge.createTask(objective, {
          evidence_scope: evidenceScope,
          workflow_mode: workflowMode,
          ...(decisionUse ? { decision_use: decisionUse } : {}),
        });
        setDraftCreated(false);
        setLocalError("");
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
            <span className="control-label">MODELING WORKFLOW</span>
            <div
              className="segmented"
              role="group"
              aria-label="Choose modeling workflow"
            >
              {(
                [
                  ["v67", "V6.7 pre-data first"],
                  ["legacy", "Legacy compatibility"],
                ] as const
              ).map(([mode, label]) => (
                <button
                  type="button"
                  key={mode}
                  className={workflowMode === mode ? "selected" : ""}
                  aria-pressed={workflowMode === mode}
                  onClick={() => {
                    setWorkflowMode(mode);
                    setLocalError("");
                  }}
                >
                  {label}
                </button>
              ))}
            </div>
            <small
              style={{
                color: "#7f8998",
                display: "block",
                fontSize: 9,
                marginTop: 8,
              }}
            >
              V6.7 freezes source and measurement semantics before S1. The
              choice is authority-bound at task creation.
            </small>
          </div>

          <div className="control-block">
            <span className="control-label">证据范围</span>
            <div className="segmented" role="group" aria-label="选择证据范围">
              {(
                [
                  ["development", "开发 / 用户数据"],
                  ["public_data", "公开注册来源"],
                ] as const
              ).map(([scope, label]) => (
                <button
                  type="button"
                  key={scope}
                  className={evidenceScope === scope ? "selected" : ""}
                  aria-pressed={evidenceScope === scope}
                  onClick={() => {
                    setEvidenceScope(scope);
                    setLocalError("");
                  }}
                >
                  {label}
                </button>
              ))}
            </div>
            <small
              style={{
                color: "#7f8998",
                display: "block",
                fontSize: 9,
                marginTop: 8,
              }}
            >
              public_data 允许后续获取已注册的 World Bank
              官方源；范围标签本身不授予科学资格。
            </small>
          </div>

          <div className="control-block">
            <span className="control-label">决策用途（可选，S0 前冻结）</span>
            <div className="segmented" role="group" aria-label="是否绑定决策用途">
              <button
                type="button"
                className={!decisionEnabled ? "selected" : ""}
                aria-pressed={!decisionEnabled}
                onClick={() => {
                  setDecisionEnabled(false);
                  setLocalError("");
                }}
              >
                仅建模报告
              </button>
              <button
                type="button"
                className={decisionEnabled ? "selected" : ""}
                aria-pressed={decisionEnabled}
                onClick={() => {
                  setDecisionEnabled(true);
                  setLocalError("");
                }}
              >
                绑定损失与阈值
              </button>
            </div>
          </div>
        </div>

        {decisionEnabled && (
          <div className="control-block" style={{ marginTop: 18 }}>
            <span className="control-label">
              V6.2 decision_use · 必须完整，不能运行后补写
            </span>
            <div style={compactFormGridStyle}>
              {[
                ["decision_id", "决策 ID", "weekly-staffing"],
                ["value_owner_ref", "价值负责人引用", "operations-owner"],
                ["action_unit", "行动单位", "staff-hours"],
              ].map(([field, label, placeholder]) => (
                <label key={field}>
                  <span className="control-label">{label}</span>
                  <input
                    style={compactInputStyle}
                    value={
                      decisionDraft[
                        field as keyof typeof decisionDraft
                      ]
                    }
                    placeholder={placeholder}
                    spellCheck={false}
                    onChange={(event) => {
                      setDecisionDraft((current) => ({
                        ...current,
                        [field]: event.target.value,
                      }));
                      setLocalError("");
                    }}
                  />
                </label>
              ))}
              {[
                ["underage_unit_cost", "少配单位代价", "2"],
                ["overage_unit_cost", "多配单位代价", "1"],
                [
                  "minimum_relative_loss_improvement",
                  "最小相对损失改善",
                  "0.05",
                ],
                [
                  "maximum_mean_normalized_regret",
                  "最大平均归一化后悔",
                  "0.20",
                ],
              ].map(([field, label, placeholder]) => (
                <label key={field}>
                  <span className="control-label">{label}</span>
                  <input
                    style={compactInputStyle}
                    type="number"
                    min="0"
                    step="any"
                    value={
                      decisionDraft[
                        field as keyof typeof decisionDraft
                      ]
                    }
                    placeholder={placeholder}
                    onChange={(event) => {
                      setDecisionDraft((current) => ({
                        ...current,
                        [field]: event.target.value,
                      }));
                      setLocalError("");
                    }}
                  />
                </label>
              ))}
            </div>
            <small
              style={{
                color: decisionComplete ? "#80d6a5" : "#e5b96b",
                display: "block",
                fontSize: 9,
                marginTop: 9,
              }}
            >
              {decisionComplete
                ? "决策用途完整；创建任务时会与目标一起冻结。"
                : "尚未完整：前端不会发送部分 decision_use。"}
            </small>
          </div>
        )}

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
                <Pill tone={s1Open ? "green" : s0Open ? "blue" : "neutral"}>
                  {s1Open
                    ? "S1 GATE OPEN"
                    : s0Open && usePredataV67
                      ? "Freeze study design before V6.7 S1"
                      : s0Open
                      ? "S0 GATE OPEN"
                      : "真实任务已冻结"}
                </Pill>
                <strong>{bridge.task.task_id}</strong>
              </div>
              <button
                className="run-agent-button"
                type="button"
                disabled={agentRunning || bridge.busy || Boolean(s1Open)}
                onClick={() => {
                  if (s0Open && usePredataV67) {
                    setSelectedStage("S1");
                    return;
                  }
                  void (s0Open ? bridge.runS1() : bridge.runS0());
                }}
              >
                {s1Open
                  ? "前往 S2 冻结 ODE 数据"
                  : agentRunning
                    ? `Codex 正在完成 ${s0Open ? "S1" : "S0"}…`
                    : s0Open
                      ? "启动并行 Codex 完成 S1"
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
            {epistemic && (
              <div className="epistemic-summary">
                <div>
                  <span>盲探索分支</span>
                  <strong>{epistemic.branch_count}</strong>
                </div>
                <div>
                  <span>来源隔离</span>
                  <strong>
                    {epistemic.independence_passed
                      ? `${epistemic.effective_independent_branches} · PASS`
                      : "FAIL"}
                  </strong>
                </div>
                <div>
                  <span>受控共享包</span>
                  <strong>{epistemic.disclosure_packet_count}</strong>
                </div>
                <div>
                  <span>跨范式迁移</span>
                  <strong>
                    {epistemic.transfer_assessment_count}/
                    {epistemic.transfer_count}
                  </strong>
                </div>
              </div>
            )}
            {portfolio && portfolio.transaction_status !== "NOT_STARTED" && (
              <div
                className="epistemic-summary"
                aria-label="V6.9 development portfolio evidence"
              >
                <div>
                  <span>V6.9 portfolio</span>
                  <strong>{portfolio.transaction_status}</strong>
                  <small>development evidence only</small>
                </div>
                <div>
                  <span>two-pack decision</span>
                  <strong>{portfolio.decision ?? "NOT_RUN"}</strong>
                </div>
                <div>
                  <span>baseline guard</span>
                  <strong>{portfolio.baseline_guard_status}</strong>
                </div>
                <div>
                  <span>selected branch</span>
                  <strong>{portfolio.selected_branch_id ?? "none"}</strong>
                </div>
              </div>
            )}
            <p>
              Graph verified:{" "}
              {bridge.task.workflow.graph_verified ? "true" : "false"} ·
              evidence scope: {bridge.taskEvidenceScope ?? "unknown"} ·
              cross-task experience:{" "}
              {epistemic?.cross_task_use_permitted ? "permitted" : "quarantined"} ·
              scientific qualification: false · real-world action: false
            </p>
          </div>
        )}

        {(localError || bridge.error) && (
          <p className="bridge-error">{localError || bridge.error}</p>
        )}
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
            ? "网页可以启动真实 S0–S1，并在兼容的正值标量 ODE 任务上执行受控 S2–S6。"
            : "启动本地 Studio Bridge 后，可执行 S0–S1 与正值标量 ODE 的 S2–S6 窄域链路。"}
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
            <small>MODELING STUDIO · V6.2</small>
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
              {bridge.connected ? "S0–S6 窄域执行可用" : "执行服务待连接"}
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
          stageStatuses={bridge.task?.workflow.stage_statuses}
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
                  setSelectedStage={setSelectedStage}
                />
              ) : selectedStage === "S1" &&
                bridge.task?.workflow.stage_statuses.S0 === "gate_open" ? (
                <S1Workspace bridge={bridge} />
              ) : ["S2", "S3", "S4", "S5", "S6"].includes(selectedStage) &&
                bridge.task?.workflow.stage_statuses.S1 === "gate_open" ? (
                <BackhalfWorkspace bridge={bridge} stageId={selectedStage} />
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
