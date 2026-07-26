"use client";

import { useCallback, useEffect, useState } from "react";

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

export function useStudioBridge() {
  const [url, setUrl] = useState("http://127.0.0.1:8765");
  const [token, setToken] = useState("");
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [task, setTask] = useState<TaskSnapshot | null>(null);

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
    async (objective: string) => {
      setBusy(true);
      setError("");
      try {
        const snapshot = await request<TaskSnapshot>("/api/v1/tasks", {
          method: "POST",
          body: JSON.stringify({
            objective,
            evidence_scope: "development",
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
    connect,
    createTask,
    runS0,
    runS1,
    refresh,
  };
}
