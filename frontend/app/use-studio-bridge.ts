"use client";

import { useCallback, useEffect, useState } from "react";

export type EvidenceStatus = "PASS" | "FAIL" | "NOT_RUN" | "HUMAN";
export type EvidenceScope = "development" | "public_data";
export type WorkflowMode = "legacy" | "v67";
export type PredataTransactionStatusV67 =
  | "NOT_STARTED"
  | "RECOVERY_PENDING"
  | "STALE_PENDING"
  | "COMPLETED"
  | "LEGACY_COMPLETED";

export type DecisionUseRequestV62 = {
  schema_version: "6.2";
  decision_id: string;
  value_owner_ref: string;
  action_unit: string;
  underage_unit_cost: number;
  overage_unit_cost: number;
  minimum_relative_loss_improvement: number;
  maximum_mean_normalized_regret: number;
};

export type CreateTaskOptions = {
  evidence_scope: EvidenceScope;
  workflow_mode: WorkflowMode;
  decision_use?: DecisionUseRequestV62;
};

export type StudioWorldBankDataRequestV62 = {
  schema_version: "6.2";
  adapter_id:
    | "scalar_autonomous_ode_v52"
    | "adaptive_positive_series_v57";
  contract_id: string;
  country_code: string;
  indicator_id: string;
  start_year: number;
  end_year: number;
  minimum_observations: number;
  state_unit: string;
  attribution: string;
  semantic_name: string;
  operational_definition: string;
  observation_time_basis: string;
  aggregation_level: string;
  fixture_only: boolean;
};

export type StudioPredataRequestSummaryV67 =
  StudioWorldBankDataRequestV62 & {
    source_contract_hash: string;
    measurement_contract_hash: string;
    protocol_hash: string;
    preparation_evidence_hash: string | null;
    intent_hash: string | null;
    completion_hash: string | null;
    capability_pack_hash: string;
  };

export type StudioEvent = {
  sequence: number;
  event_type: string;
  status: "accepted" | "running" | "succeeded" | "failed" | "blocked";
  message: string;
  details: Record<string, unknown>;
  recorded_at: string;
  event_hash: string;
};

export type TaskSnapshot = {
  status: "success";
  task_id: string;
  objective: string;
  evidence_scope: EvidenceScope;
  workflow: {
    graph_verified: boolean;
    stage_statuses: Record<string, string>;
    frontier_stages: string[];
    scientific_qualification_granted: false;
    real_world_action_authorized: false;
  };
  activity: "idle" | "accepted" | "running" | "succeeded" | "failed" | "blocked";
  events: StudioEvent[];
  epistemic: {
    schema_version: "5.8";
    graph_hash: string;
    knowledge_unit_count: number;
    branch_count: number;
    effective_independent_branches: number;
    independence_passed: boolean;
    independence_scope: "origin_separation_only";
    scientific_independence_established: false;
    disclosure_packet_count: number;
    transfer_count: number;
    transfer_assessment_count: number;
    cross_task_experience_count: number;
    cross_task_use_permitted: boolean;
  } | null;
  backhalf: {
    schema_version: "6.0";
    adapter_id:
      | "scalar_autonomous_ode_v52"
      | "adaptive_positive_series_v57";
    data_received: boolean;
    workflow_complete: boolean;
    selected_scientific_family: string | null;
    selected_branch: string | null;
    recovery_triggered: boolean;
    level_statuses: Record<string, string>;
    scientific_acceptance: boolean;
    fixture_only: boolean | null;
    source_integrity_status?: EvidenceStatus;
    scientific_provenance_status?: EvidenceStatus;
    source_stage_admission_status?: EvidenceStatus;
    rolling_confirmation_admission_status?: EvidenceStatus;
    rolling_confirmation_status?: EvidenceStatus;
    decision_evidence_admission_status?: EvidenceStatus;
    decision_evidence_status?: EvidenceStatus;
    scientific_decision_status?: EvidenceStatus;
    executable_candidate_admission_status?: EvidenceStatus;
    executable_candidate_status?: EvidenceStatus;
    scientific_qualification_granted: false;
    real_world_action_authorized: false;
  };
  recovery: {
    schema_version: "6.0";
    policy_hash: string;
    scientific_attempts_started: number;
    attempt_budget_remaining: number;
    same_attempt_retries: number;
    distinct_failure_signatures: number;
    stopped: boolean;
    stop_reason: string | null;
    human_required: boolean;
    human_reason: string | null;
    last_action:
      | "RETRY"
      | "PATCH"
      | "BRANCH"
      | "ACQUIRE_DATA"
      | "ABSTAIN"
      | "HUMAN"
      | null;
    last_revoke_from: string | null;
    event_count: number;
    last_event_hash: string | null;
    private_adaptive_feedback_permitted: false;
    scientific_qualification_granted: false;
    real_world_action_authorized: false;
  };
  scientific_success: {
    schema_version: "6.1";
    evaluated: boolean;
    claim_kind: "predictive";
    local_predictive_gate_status: EvidenceStatus;
    scientific_success_status: EvidenceStatus;
    claim_ceiling:
      | "no_scientific_claim"
      | "workflow_integrity_only"
      | "fixture_protocol_only"
      | "local_retrospective_adapter_evidence"
      | "local_leakage_safe_predictive_evidence"
      | "externally_qualified_predictive_evidence";
    fixture_only?: boolean;
    dimensions: Record<
      string,
      {
        status: EvidenceStatus;
        required_for_claim: boolean;
        reason_codes: string[];
        metrics: Record<string, number | boolean | null>;
      }
    >;
    confirmation: {
      status: EvidenceStatus;
      completed_fold_count: number;
      requested_fold_count: number;
      selected_model_ids: string[];
      metrics: Record<string, number | null>;
      reason_codes: string[];
    } | null;
    report_hash?: string;
    scientific_qualification_granted: false;
    real_world_action_authorized: false;
  };
  scientific_closure?: {
    schema_version?: string;
    evaluated?: boolean;
    source_integrity_status?: EvidenceStatus;
    scientific_provenance_status?: EvidenceStatus;
    decision_evidence_status?: EvidenceStatus;
    scientific_decision_status?: EvidenceStatus;
    stage_admission_status?: EvidenceStatus;
    closure_verification_status?: EvidenceStatus;
    local_evidence_status?: EvidenceStatus;
    scientific_closure_status?: EvidenceStatus;
    claim_ceiling?:
      | "no_scientific_claim"
      | "workflow_integrity_only"
      | "fixture_protocol_only"
      | "local_retrospective_adapter_evidence"
      | "local_leakage_safe_predictive_evidence"
      | "externally_qualified_predictive_evidence";
    fixture_only?: boolean;
    dimensions?:
      | Record<
          string,
          {
            status: EvidenceStatus;
            required_for_claim: boolean;
            reason_codes: string[];
          }
        >
      | {
          dimension_id: string;
          status: EvidenceStatus;
          required_for_claim: boolean;
          reason_codes: string[];
        }[];
    scientific_qualification_granted?: false;
    real_world_action_authorized?: false;
  } | null;
  predata_v67: {
    schema_version: "6.7";
    workflow_mode: WorkflowMode;
    available: boolean;
    prepared: boolean;
    required_before_v67_s1: boolean;
    transaction_status: PredataTransactionStatusV67;
    recovery_available: boolean;
    request_summary: StudioPredataRequestSummaryV67 | null;
    source_contract_hash: string | null;
    measurement_contract_hash: string | null;
    protocol_hash: string | null;
    intent_hash: string | null;
    completion_hash: string | null;
    observation_values_included: false;
    private_acceptance_data_included: false;
    scientific_qualification_granted: false;
    real_world_action_authorized: false;
  };
  portfolio_v69?: {
    schema_version: "6.9";
    development_only: true;
    available: boolean;
    transaction_status:
      | "NOT_STARTED"
      | "PREPARED"
      | "DATA_READY"
      | "RUN_PENDING"
      | "COMPLETED"
      | "STALE_PENDING";
    recovery_available: boolean;
    protocol_hash: string | null;
    snapshot_hash: string | null;
    outer_origin_plan_hash: string | null;
    branch_statuses: Record<string, "PASS" | "FAIL" | "NOT_RUN">;
    evaluation_hashes: Record<string, string>;
    decision: "SELECT" | "ABSTAIN" | null;
    selected_branch_id: string | null;
    decision_hash: string | null;
    run_hash: string | null;
    baseline_guard_status: "PASS" | "FAIL" | "NOT_RUN";
    persistence_relative_improvement: number | null;
    engineering_status:
      | "NOT_STARTED"
      | "PREPARED"
      | "DATA_READY"
      | "RUN_PENDING"
      | "COMPLETED"
      | "STALE_PENDING";
    scientific_evidence_status: "NOT_RUN";
    claim_ceiling: "development_protocol_only";
    problem_signature_source: "caller_selected_v69_narrow_lane";
    derived_from_s0_typed_problem_signature: false;
    s1_s6_gates_touched: false;
    scientific_qualification_granted: false;
    real_world_action_authorized: false;
  };
  next_valid_actions: string[];
  scientific_qualification_granted: false;
  real_world_action_authorized: false;
};

function normalizedLoopbackUrl(value: string): string {
  const url = new URL(value.trim());
  if (url.protocol !== "http:") {
    throw new Error("本地执行桥目前只接受 http loopback 地址。");
  }
  if (!["127.0.0.1", "localhost", "[::1]"].includes(url.hostname)) {
    throw new Error("为保护桥接令牌，只允许连接本机 loopback 地址。");
  }
  return url.origin;
}

async function parseResponse<T>(response: Response): Promise<T> {
  const payload = (await response.json()) as {
    status?: string;
    message?: string;
  };
  if (!response.ok) {
    throw new Error(payload.message || `执行桥返回 ${response.status}`);
  }
  return payload as T;
}

function assertCompleteDecisionUse(
  decisionUse: DecisionUseRequestV62 | undefined,
): void {
  if (!decisionUse) return;
  const requiredText = [
    decisionUse.decision_id,
    decisionUse.value_owner_ref,
    decisionUse.action_unit,
  ];
  if (requiredText.some((value) => !value.trim())) {
    throw new Error("决策用途已启用，但决策 ID、价值负责人和行动单位未填写完整。");
  }
  if (
    !Number.isFinite(decisionUse.underage_unit_cost) ||
    decisionUse.underage_unit_cost <= 0 ||
    !Number.isFinite(decisionUse.overage_unit_cost) ||
    decisionUse.overage_unit_cost <= 0
  ) {
    throw new Error("少配与多配单位代价必须是大于 0 的有限数值。");
  }
  if (
    !Number.isFinite(decisionUse.minimum_relative_loss_improvement) ||
    decisionUse.minimum_relative_loss_improvement < 0 ||
    decisionUse.minimum_relative_loss_improvement > 1
  ) {
    throw new Error("最小相对损失改善阈值必须位于 0 到 1 之间。");
  }
  if (
    !Number.isFinite(decisionUse.maximum_mean_normalized_regret) ||
    decisionUse.maximum_mean_normalized_regret <= 0 ||
    decisionUse.maximum_mean_normalized_regret > 1
  ) {
    throw new Error("最大平均归一化后悔阈值必须大于 0 且不超过 1。");
  }
}

function preparedPredataRequestFromSnapshot(
  snapshot: TaskSnapshot,
): StudioWorldBankDataRequestV62 | null {
  const summary = snapshot.predata_v67.request_summary;
  if (
    !summary ||
    (snapshot.predata_v67.prepared !== true &&
      snapshot.predata_v67.transaction_status !== "RECOVERY_PENDING" &&
      snapshot.predata_v67.transaction_status !== "STALE_PENDING")
  ) {
    return null;
  }
  return {
    schema_version: summary.schema_version,
    adapter_id: summary.adapter_id,
    contract_id: summary.contract_id,
    country_code: summary.country_code,
    indicator_id: summary.indicator_id,
    start_year: summary.start_year,
    end_year: summary.end_year,
    minimum_observations: summary.minimum_observations,
    state_unit: summary.state_unit,
    attribution: summary.attribution,
    semantic_name: summary.semantic_name,
    operational_definition: summary.operational_definition,
    observation_time_basis: summary.observation_time_basis,
    aggregation_level: summary.aggregation_level,
    fixture_only: summary.fixture_only,
  };
}

export function useStudioBridge() {
  const [url, setUrl] = useState("http://127.0.0.1:8765");
  const [token, setToken] = useState("");
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [task, setTask] = useState<TaskSnapshot | null>(null);
  const taskEvidenceScope = task?.evidence_scope ?? null;
  const preparedPredataRequest = task
    ? preparedPredataRequestFromSnapshot(task)
    : null;

  const request = useCallback(
    async <T,>(
      path: string,
      init: RequestInit = {},
      requiresToken = true,
    ): Promise<T> => {
      const base = normalizedLoopbackUrl(url);
      const headers = new Headers(init.headers);
      headers.set("Content-Type", "application/json");
      if (requiresToken) {
        if (token.length < 24) {
          throw new Error("请输入本地执行桥启动时使用的令牌。");
        }
        headers.set("X-FMA-Bridge-Token", token);
      }
      const response = await fetch(`${base}${path}`, {
        ...init,
        headers,
        cache: "no-store",
      });
      return parseResponse<T>(response);
    },
    [token, url],
  );

  const connect = useCallback(async () => {
    setBusy(true);
    setError("");
    try {
      const health = await request<{
        status: string;
        service: string;
        authority_key_exposed: false;
      }>("/api/v1/health", { method: "GET" }, false);
      if (
        health.status !== "ok" ||
        health.service !== "fma-studio-bridge" ||
        health.authority_key_exposed !== false
      ) {
        throw new Error("目标服务不是可信的 FMA Studio Bridge。");
      }
      setConnected(true);
    } catch (reason) {
      setConnected(false);
      setError(reason instanceof Error ? reason.message : "连接失败");
    } finally {
      setBusy(false);
    }
  }, [request]);

  const createTask = useCallback(
    async (
      objective: string,
      options: CreateTaskOptions = {
        evidence_scope: "development",
        workflow_mode: "legacy",
      },
    ) => {
      setBusy(true);
      setError("");
      try {
        assertCompleteDecisionUse(options.decision_use);
        const snapshot = await request<TaskSnapshot>("/api/v1/tasks", {
          method: "POST",
          body: JSON.stringify({
            objective,
            evidence_scope: options.evidence_scope,
            workflow_mode: options.workflow_mode,
            ...(options.decision_use
              ? { decision_use: options.decision_use }
              : {}),
          }),
        });
        setTask(snapshot);
        return snapshot;
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : "任务创建失败");
        throw reason;
      } finally {
        setBusy(false);
      }
    },
    [request],
  );

  const runS0 = useCallback(async () => {
    if (!task) throw new Error("请先创建真实 FMA 任务。");
    setBusy(true);
    setError("");
    try {
      const snapshot = await request<TaskSnapshot>(
        `/api/v1/tasks/${encodeURIComponent(task.task_id)}/run-s0`,
        { method: "POST", body: "{}" },
      );
      setTask(snapshot);
      return snapshot;
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "S0 启动失败");
      throw reason;
    } finally {
      setBusy(false);
    }
  }, [request, task]);

  const preparePredata = useCallback(
    async (payload: StudioWorldBankDataRequestV62) => {
      if (!task) {
        throw new Error(
          "Create a task and complete S0 before preparing the V6.7 pre-data contracts.",
        );
      }
      setBusy(true);
      setError("");
      try {
        const snapshot = await request<TaskSnapshot>(
          `/api/v1/tasks/${encodeURIComponent(task.task_id)}/prepare-predata`,
          { method: "POST", body: JSON.stringify(payload) },
        );
        setTask(snapshot);
        return snapshot;
      } catch (reason) {
        setError(
          reason instanceof Error
            ? reason.message
            : "V6.7 pre-data preparation failed",
        );
        throw reason;
      } finally {
        setBusy(false);
      }
    },
    [request, task],
  );

  const reconcilePredata = useCallback(async () => {
    if (!task) {
      throw new Error("Create a task before reconciling its V6.7 pre-data transaction.");
    }
    setBusy(true);
    setError("");
    try {
      const snapshot = await request<TaskSnapshot>(
        `/api/v1/tasks/${encodeURIComponent(task.task_id)}/reconcile-predata`,
        { method: "POST" },
      );
      setTask(snapshot);
      return snapshot;
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "V6.7 pre-data reconciliation failed",
      );
      throw reason;
    } finally {
      setBusy(false);
    }
  }, [request, task]);

  const runS1 = useCallback(async () => {
    if (!task) throw new Error("请先创建并完成真实 FMA S0。");
    setBusy(true);
    setError("");
    try {
      const snapshot = await request<TaskSnapshot>(
        `/api/v1/tasks/${encodeURIComponent(task.task_id)}/run-s1`,
        { method: "POST", body: "{}" },
      );
      setTask(snapshot);
      return snapshot;
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "S1 启动失败");
      throw reason;
    } finally {
      setBusy(false);
    }
  }, [request, task]);

  const ingestOdeData = useCallback(
    async (payload: Record<string, unknown>) => {
      if (!task) throw new Error("请先完成 S0–S1，再冻结 ODE 数据。");
      setBusy(true);
      setError("");
      try {
        const snapshot = await request<TaskSnapshot>(
          `/api/v1/tasks/${encodeURIComponent(task.task_id)}/data/ode`,
          { method: "POST", body: JSON.stringify(payload) },
        );
        setTask(snapshot);
        return snapshot;
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : "ODE 数据冻结失败");
        throw reason;
      } finally {
        setBusy(false);
      }
    },
    [request, task],
  );

  const ingestWorldBankData = useCallback(
    async (payload: StudioWorldBankDataRequestV62) => {
      if (!task) {
        throw new Error("请先完成 S0–S1，再登记 World Bank 官方数据源。");
      }
      setBusy(true);
      setError("");
      try {
        const snapshot = await request<TaskSnapshot>(
          `/api/v1/tasks/${encodeURIComponent(task.task_id)}/data/world-bank`,
          { method: "POST", body: JSON.stringify(payload) },
        );
        setTask(snapshot);
        return snapshot;
      } catch (reason) {
        setError(
          reason instanceof Error
            ? reason.message
            : "World Bank 官方数据源登记失败",
        );
        throw reason;
      } finally {
        setBusy(false);
      }
    },
    [request, task],
  );

  const runBackhalf = useCallback(async () => {
    if (!task) throw new Error("请先完成 S0–S1 并冻结 ODE 数据。");
    setBusy(true);
    setError("");
    try {
      const snapshot = await request<TaskSnapshot>(
        `/api/v1/tasks/${encodeURIComponent(task.task_id)}/run-backhalf`,
        { method: "POST", body: "{}" },
      );
      setTask(snapshot);
      return snapshot;
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "S2–S6 启动失败");
      throw reason;
    } finally {
      setBusy(false);
    }
  }, [request, task]);

  const refresh = useCallback(async () => {
    if (!task) return null;
    try {
      const snapshot = await request<TaskSnapshot>(
        `/api/v1/tasks/${encodeURIComponent(task.task_id)}`,
        { method: "GET" },
      );
      setTask(snapshot);
      return snapshot;
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "状态刷新失败");
      return null;
    }
  }, [request, task]);

  useEffect(() => {
    if (!task || !["accepted", "running"].includes(task.activity)) return;
    const timer = window.setInterval(() => {
      void refresh();
    }, 2000);
    return () => window.clearInterval(timer);
  }, [refresh, task]);

  return {
    url,
    setUrl,
    token,
    setToken,
    connected,
    busy,
    error,
    task,
    taskEvidenceScope,
    preparedPredataRequest,
    connect,
    createTask,
    runS0,
    preparePredata,
    reconcilePredata,
    runS1,
    ingestOdeData,
    ingestWorldBankData,
    runBackhalf,
    refresh,
  };
}
