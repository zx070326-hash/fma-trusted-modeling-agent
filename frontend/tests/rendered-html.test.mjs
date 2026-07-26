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

test("server-renders the governed FMA public result", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>FMA · 数学建模智能体控制台<\/title>/i);
  assert.match(html, /模型不是一次命中/);
  assert.match(html, /L0–L4/);
  assert.match(html, /PRIVATE EVALUATION/);
  assert.match(html, /BLOCKED/);
  assert.match(html, /对数漂移/);
  assert.match(html, /aria-pressed="true"/);
  assert.match(html, /https:\/\/fma\.example\/og\.png/);
  assert.doesNotMatch(html, /Your site is taking shape|Building your site/);
});

test("keeps evidence, graph, and authority data explicit in source", async () => {
  const [page, data, layout] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/modeling-data.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
  ]);

  assert.match(page, /<canvas/);
  assert.match(page, /aria-pressed=\{selectedNode === item\.id\}/);
  assert.match(page, /PRIVATE EVALUATION/);
  assert.match(data, /\{ label: "科学资格", value: "FALSE" \}/);
  assert.match(page, /公开通过，不代表外部资格/);
  assert.match(page, /EXTERNAL HOST · NOT RUN/);
  assert.match(layout, /generateMetadata/);
  assert.doesNotMatch(page, /react-loading-skeleton|SkeletonPreview/);
});
