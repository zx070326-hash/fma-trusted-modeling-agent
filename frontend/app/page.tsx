"use client";

import { useEffect, useRef, useState } from "react";
import {
  candidates,
  evidenceLevels,
  forecastSeries,
  graphNodes,
  task,
} from "./modeling-data";

function StatusMark({ status }: { status: string }) {
  return (
    <span className={`status-mark status-${status.toLowerCase()}`}>
      <span className="status-dot" aria-hidden="true" />
      {status}
    </span>
  );
}

function ForecastChart() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const scale = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = width * scale;
    canvas.height = height * scale;
    context.scale(scale, scale);
    context.clearRect(0, 0, width, height);

    const pad = { top: 18, right: 18, bottom: 28, left: 42 };
    const plotWidth = width - pad.left - pad.right;
    const plotHeight = height - pad.top - pad.bottom;
    const values = forecastSeries.map((point) => point.value);
    const minimum = Math.min(...values) * 0.93;
    const maximum = Math.max(...values) * 1.04;
    const x = (index: number) =>
      pad.left + (index / (forecastSeries.length - 1)) * plotWidth;
    const y = (value: number) =>
      pad.top + (1 - (value - minimum) / (maximum - minimum)) * plotHeight;

    context.strokeStyle = "rgba(255,255,255,.08)";
    context.lineWidth = 1;
    context.fillStyle = "rgba(207,216,210,.52)";
    context.font = "11px ui-monospace, SFMono-Regular, monospace";
    context.textAlign = "right";
    for (let index = 0; index <= 3; index += 1) {
      const value = minimum + ((maximum - minimum) * index) / 3;
      const lineY = y(value);
      context.beginPath();
      context.moveTo(pad.left, lineY);
      context.lineTo(width - pad.right, lineY);
      context.stroke();
      context.fillText(value.toFixed(0), pad.left - 9, lineY + 4);
    }

    const observedCount = forecastSeries.filter(
      (point) => point.kind === "observed",
    ).length;
    const splitX = x(observedCount - 0.5);
    context.setLineDash([4, 5]);
    context.strokeStyle = "rgba(255,202,106,.4)";
    context.beginPath();
    context.moveTo(splitX, pad.top);
    context.lineTo(splitX, height - pad.bottom);
    context.stroke();
    context.setLineDash([]);

    const drawLine = (
      start: number,
      end: number,
      color: string,
      dashed = false,
    ) => {
      context.strokeStyle = color;
      context.lineWidth = 2.4;
      context.lineJoin = "round";
      context.lineCap = "round";
      context.setLineDash(dashed ? [6, 5] : []);
      context.beginPath();
      for (let index = start; index <= end; index += 1) {
        const point = forecastSeries[index];
        if (index === start) context.moveTo(x(index), y(point.value));
        else context.lineTo(x(index), y(point.value));
      }
      context.stroke();
      context.setLineDash([]);
    };

    drawLine(0, observedCount - 1, "#62e6a7");
    drawLine(observedCount - 1, forecastSeries.length - 1, "#ffca6a", true);

    forecastSeries.slice(observedCount).forEach((point, offset) => {
      const index = observedCount + offset;
      context.fillStyle = "#ffca6a";
      context.beginPath();
      context.arc(x(index), y(point.value), 3.2, 0, Math.PI * 2);
      context.fill();
    });

    context.fillStyle = "rgba(255,202,106,.78)";
    context.textAlign = "left";
    context.fillText("预测注册点", splitX + 8, pad.top + 11);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="forecast-canvas"
      aria-label="公开观测与四步注册预测折线图"
    />
  );
}

export default function Home() {
  const [selectedNode, setSelectedNode] = useState("recovery");
  const [showMetrics, setShowMetrics] = useState(false);
  const node = graphNodes.find((item) => item.id === selectedNode)!;

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#overview" aria-label="FMA 首页">
          <span className="brand-glyph">F</span>
          <span>
            <strong>FMA</strong>
            <small>GRAPH-NATIVE · V5.7</small>
          </span>
        </a>
        <div className="topbar-center">
          <span className="connection-dot" />
          建模内核已连接
        </div>
        <div className="topbar-actions">
          <span className="task-chip">I36 · 未见任务</span>
          <a className="evidence-link" href="#evidence">
            查看证据
            <span aria-hidden="true">↗</span>
          </a>
        </div>
      </header>

      <aside className="sidebar" aria-label="控制台导航">
        <nav>
          <p className="nav-label">工作区</p>
          <a className="nav-item active" href="#overview">
            <span>01</span>运行总览
          </a>
          <a className="nav-item" href="#graph">
            <span>02</span>模型图
          </a>
          <a className="nav-item" href="#forecast">
            <span>03</span>预测
          </a>
          <a className="nav-item" href="#evidence">
            <span>04</span>证据层
          </a>
          <a className="nav-item" href="#boundary">
            <span>05</span>权限边界
          </a>
        </nav>

        <div className="sidebar-meta">
          <p>当前任务</p>
          <strong>{task.shortId}</strong>
          <span>公开运行 · 已冻结</span>
          <div className="mini-rule" />
          <p>实现提交</p>
          <code>{task.commit}</code>
        </div>
      </aside>

      <main>
        <section className="hero" id="overview">
          <div className="eyebrow">
            <span>PUBLIC GATE</span>
            <StatusMark status="ELIGIBLE" />
          </div>
          <div className="hero-grid">
            <div>
              <h1>
                模型不是一次命中，
                <br />
                而是在图中恢复。
              </h1>
              <p className="hero-copy">
                首次真实双重未见任务中，初始 ODE 在 L3
                失败。系统没有放宽标准，而是切换表征，筛出可复现的对数增长模型并注册四步预测。
              </p>
            </div>
            <div className="hero-verdict">
              <div className="verdict-ring">
                <strong>5/5</strong>
                <span>证据层通过</span>
              </div>
              <p>
                公开科学接受
                <small>不等于外部科学资格</small>
              </p>
            </div>
          </div>

          <div className="metric-strip">
            <div>
              <span>公开门</span>
              <strong className="accent-green">ELIGIBLE</strong>
            </div>
            <div>
              <span>新鲜重放</span>
              <strong>2 / 2</strong>
            </div>
            <div>
              <span>可接受恢复</span>
              <strong>1 / 2</strong>
            </div>
            <div>
              <span>私测消耗</span>
              <strong>0 / 1</strong>
            </div>
            <div>
              <span>现实行动</span>
              <strong className="muted-value">未授权</strong>
            </div>
          </div>
        </section>

        <section className="panel graph-panel" id="graph">
          <div className="section-heading">
            <div>
              <span className="section-index">02 / MODEL GRAPH</span>
              <h2>贯穿全链路的候选图</h2>
            </div>
            <p>点击节点查看决策依据</p>
          </div>

          <div className="graph-flow" role="list" aria-label="模型执行图">
            {graphNodes.map((item, index) => (
              <div className="graph-step-wrap" key={item.id}>
                <button
                  className={`graph-node graph-${item.tone} ${
                    selectedNode === item.id ? "selected" : ""
                  }`}
                  onClick={() => setSelectedNode(item.id)}
                  type="button"
                  aria-pressed={selectedNode === item.id}
                >
                  <span className="node-index">
                    {String(index + 1).padStart(2, "0")}
                  </span>
                  <strong>{item.label}</strong>
                  <small>{item.status}</small>
                </button>
                {index < graphNodes.length - 1 && (
                  <span className="graph-arrow" aria-hidden="true">
                    →
                  </span>
                )}
              </div>
            ))}
          </div>

          <div className={`node-inspector inspector-${node.tone}`}>
            <div>
              <span>SELECTED NODE · {node.status}</span>
              <h3>{node.title}</h3>
            </div>
            <p>{node.description}</p>
            <dl>
              {node.facts.map((fact) => (
                <div key={fact.label}>
                  <dt>{fact.label}</dt>
                  <dd>{fact.value}</dd>
                </div>
              ))}
            </dl>
          </div>
        </section>

        <div className="two-column">
          <section className="panel forecast-panel" id="forecast">
            <div className="section-heading compact">
              <div>
                <span className="section-index">03 / FORECAST</span>
                <h2>公开序列与注册预测</h2>
              </div>
              <div className="legend" aria-label="图例">
                <span className="legend-observed">公开观测</span>
                <span className="legend-predicted">冻结预测</span>
              </div>
            </div>
            <ForecastChart />
            <div className="forecast-values">
              {task.predictions.map((value, index) => (
                <div key={value}>
                  <span>H{index + 1}</span>
                  <strong>{value.toFixed(2)}</strong>
                </div>
              ))}
            </div>
            <p className="chart-note">
              纵轴为盲化正值指数；私有目标未读取，预测哈希已冻结。
            </p>
          </section>

          <section className="panel candidates-panel">
            <div className="section-heading compact">
              <div>
                <span className="section-index">CANDIDATE ROUTER</span>
                <h2>为什么不是最低误差者</h2>
              </div>
            </div>
            <div className="candidate-list">
              {candidates.map((candidate) => (
                <article
                  className={`candidate-card ${
                    candidate.selected ? "candidate-selected" : ""
                  }`}
                  key={candidate.id}
                >
                  <div>
                    <span>{candidate.family}</span>
                    <strong>{candidate.name}</strong>
                  </div>
                  <StatusMark status={candidate.status} />
                  <dl>
                    <div>
                      <dt>验证相对 RMSE</dt>
                      <dd>{candidate.rmse}</dd>
                    </div>
                    <div>
                      <dt>相对持久性提升</dt>
                      <dd>{candidate.improvement}</dd>
                    </div>
                  </dl>
                  <p>{candidate.reason}</p>
                </article>
              ))}
            </div>
          </section>
        </div>

        <section className="panel evidence-panel" id="evidence">
          <div className="section-heading">
            <div>
              <span className="section-index">04 / EVIDENCE</span>
              <h2>不是一枚总分，而是五层可追溯证据</h2>
            </div>
            <button
              className="text-button"
              type="button"
              onClick={() => setShowMetrics((value) => !value)}
              aria-expanded={showMetrics}
            >
              {showMetrics ? "收起关键指标" : "展开关键指标"}
              <span aria-hidden="true">{showMetrics ? "−" : "+"}</span>
            </button>
          </div>

          <div className="evidence-table" role="table">
            {evidenceLevels.map((level) => (
              <article className="evidence-row" role="row" key={level.level}>
                <div className="level-token">{level.level}</div>
                <div className="evidence-copy">
                  <strong>{level.title}</strong>
                  <p>{level.description}</p>
                  {showMetrics && (
                    <div className="evidence-metrics">
                      {level.metrics.map((metric) => (
                        <span key={metric}>{metric}</span>
                      ))}
                    </div>
                  )}
                </div>
                <StatusMark status="PASS" />
              </article>
            ))}
          </div>
        </section>

        <section className="boundary" id="boundary">
          <div className="boundary-icon" aria-hidden="true">
            !
          </div>
          <div>
            <span className="section-index">05 / AUTHORITY BOUNDARY</span>
            <h2>公开通过，不代表外部资格。</h2>
            <p>
              预测已经注册，但私有评估必须由独立信任节点完成。同机新会话只能提供上下文隔离，不能提供密钥、权限与管理权隔离。
            </p>
          </div>
          <div className="boundary-status">
            <span>PRIVATE EVALUATION</span>
            <strong>BLOCKED</strong>
            <small>EXTERNAL HOST · NOT RUN</small>
          </div>
        </section>

        <footer>
          <span>FMA · TRUSTWORTHY MATHEMATICAL MODELLING KERNEL</span>
          <span>任务 {task.shortId} · 所有数值均来自冻结公开证据</span>
        </footer>
      </main>
    </div>
  );
}
