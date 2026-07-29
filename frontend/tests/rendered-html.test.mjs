import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("https://fma.example/", {
      headers: {
        accept: "text/html",
        host: "fma.example",
        "x-forwarded-host": "fma.example",
        "x-forwarded-proto": "https",
      },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the real modeling workbench", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>FMA · 真实数学建模工作台<\/title>/i);
  assert.match(html, /从真实问题开始/);
  assert.match(html, /你现在想解决什么/);
  assert.match(html, /系统边界是什么/);
  assert.match(html, /Graph 贯穿其中/);
  assert.match(html, /执行服务待连接/);
  assert.match(html, /本地执行桥/);
  assert.match(html, /新建真实任务/);
  assert.match(html, /https:\/\/fma\.example\/og-v59\.png/);
  assert.doesNotMatch(html, /Your site is taking shape|Building your site/);
});

test("keeps workflow, execution truth, and authority explicit in source", async () => {
  const [page, data, layout, bridge] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/modeling-data.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/use-studio-bridge.ts", import.meta.url), "utf8"),
  ]);

  assert.match(page, /建立本地任务草案/);
  assert.match(page, /本页面没有伪造这次模型调用/);
  assert.match(page, /启动 Agent · 待接入执行服务/);
  assert.match(page, /创建真实 FMA 任务/);
  assert.match(page, /启动 Codex 完成 S0/);
  assert.match(page, /启动并行 Codex 完成 S1/);
  assert.match(page, /启动 Graph-native S1/);
  assert.match(page, /盲探索分支/);
  assert.match(page, /共享知识不等于共享结论/);
  assert.match(page, /cross-task experience/);
  assert.match(page, /V6\.9 development portfolio evidence/);
  assert.match(page, /baseline guard/);
  assert.match(page, /development evidence only/);
  assert.match(page, /尚未完整：前端不会发送部分 decision_use/);
  assert.match(page, /public_data/);
  assert.match(page, /World Bank 注册源/);
  assert.match(page, /fixture_only（默认 false；仅 development/);
  assert.match(page, /typed semantics admission/);
  assert.match(page, /executed semantics/);
  assert.match(page, /V6\.7 pre-data first/);
  assert.match(page, /Prepare V6\.7 pre-data contracts/);
  assert.match(page, /V6\.7 RECOVERY PENDING/);
  assert.match(page, /Resume exact pre-data transaction/);
  assert.match(page, /V6\.7 STALE S0 INTENT/);
  assert.match(page, /Stale pre-data intent requires graph recovery/);
  assert.match(page, /predataStalePending/);
  assert.match(page, /bound to an obsolete S0 gate/);
  assert.match(page, /predataRequestFrozen/);
  assert.match(page, /bridge\.reconcilePredata\(\)/);
  assert.match(page, /usePredataV67 && !predataPrepared/);
  assert.match(page, /predata_v67\.workflow_mode === "v67"/);
  assert.doesNotMatch(
    page,
    /next_valid_actions\.includes\("prepare_predata_v67"\)/,
  );
  assert.match(page, /type="file"/);
  assert.match(page, /AUTHORITY BOUNDARY/);
  assert.match(data, /\{ label: "科学资格", value: "FALSE" \}/);
  assert.match(data, /id: "S0"/);
  assert.match(data, /id: "S6"/);
  assert.match(data, /role: "Harness"/);
  assert.match(bridge, /X-FMA-Bridge-Token/);
  assert.match(bridge, /run-s1/);
  assert.match(bridge, /prepare-predata/);
  assert.match(bridge, /preparePredata/);
  assert.match(bridge, /reconcile-predata/);
  assert.match(bridge, /reconcilePredata/);
  assert.match(bridge, /RECOVERY_PENDING/);
  assert.match(bridge, /STALE_PENDING/);
  assert.match(bridge, /portfolio_v69/);
  assert.match(bridge, /s1_s6_gates_touched: false/);
  assert.match(bridge, /run-backhalf/);
  assert.match(bridge, /data\/ode/);
  assert.match(bridge, /data\/world-bank/);
  assert.match(bridge, /assertCompleteDecisionUse/);
  assert.match(bridge, /workflow_mode: options\.workflow_mode/);
  assert.match(bridge, /preparedPredataRequestFromSnapshot\(task\)/);
  assert.doesNotMatch(
    bridge,
    /useState<StudioWorldBankDataRequestV62 \| null>/,
  );
  assert.match(bridge, /只允许连接本机 loopback 地址/);
  assert.doesNotMatch(bridge, /localStorage|sessionStorage/);
  assert.match(layout, /generateMetadata/);
  assert.doesNotMatch(page, /react-loading-skeleton|SkeletonPreview/);
});
