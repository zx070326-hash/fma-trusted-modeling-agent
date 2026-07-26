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
  assert.match(html, /执行服务待接入/);
  assert.match(html, /新建真实任务/);
  assert.match(html, /https:\/\/fma\.example\/og\.png/);
  assert.doesNotMatch(html, /Your site is taking shape|Building your site/);
});

test("keeps workflow, execution truth, and authority explicit in source", async () => {
  const [page, data, layout] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/modeling-data.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
  ]);

  assert.match(page, /建立本地任务草案/);
  assert.match(page, /本页面没有伪造这次模型调用/);
  assert.match(page, /启动 Agent · 待接入执行服务/);
  assert.match(page, /type="file"/);
  assert.match(page, /AUTHORITY BOUNDARY/);
  assert.match(data, /\{ label: "科学资格", value: "FALSE" \}/);
  assert.match(data, /id: "S0"/);
  assert.match(data, /id: "S6"/);
  assert.match(data, /role: "Harness"/);
  assert.match(layout, /generateMetadata/);
  assert.doesNotMatch(page, /react-loading-skeleton|SkeletonPreview/);
});
