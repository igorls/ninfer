#pragma once

#include <string_view>

namespace ninfer::supervisor {

inline constexpr std::string_view kDashboardHtml = R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NInfer Supervisor // Mission Control</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {
    --bg-base: #080a0f;
    --bg-surface: #0e121a;
    --bg-card: #131824;
    --bg-card-hover: #171e2e;
    --bg-subtle: #1a2233;
    --bg-input: #0a0d14;
    --border-dim: #1c2436;
    --border-line: #263147;
    --border-highlight: #3b4968;
    --text-main: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --text-dim: #475569;
    --ok: #10b981;
    --ok-glow: rgba(16, 185, 129, 0.15);
    --bad: #ef4444;
    --bad-glow: rgba(239, 68, 68, 0.15);
    --warn: #f59e0b;
    --warn-glow: rgba(245, 158, 11, 0.15);
    --accent: #38bdf8;
    --accent-glow: rgba(56, 189, 248, 0.15);
    --purple: #a855f7;
    --font-mono: "Cascadia Code", "JetBrains Mono", Consolas, "SF Mono", monospace;
    --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg-base);
    color: var(--text-main);
    font-family: var(--font-sans);
    font-size: 13px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
    min-height: 100vh;
    padding-bottom: 40px;
  }
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg-base); }
  ::-webkit-scrollbar-thumb { background: var(--border-line); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--border-highlight); }

  /* Top Navigation Bar */
  header {
    background: var(--bg-surface);
    border-bottom: 1px solid var(--border-dim);
    padding: 12px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(12px);
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .brand-logo {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid var(--accent);
    border-radius: 6px;
    color: var(--accent);
    font-family: var(--font-mono);
    font-weight: 800;
    font-size: 14px;
    box-shadow: 0 0 12px var(--accent-glow);
  }
  .brand-title {
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-main);
  }
  .brand-sub {
    font-size: 11px;
    color: var(--text-muted);
    font-family: var(--font-mono);
  }
  .header-badges {
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .pulse-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 9999px;
    background: var(--bg-subtle);
    border: 1px solid var(--border-dim);
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--text-secondary);
  }
  .pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--ok);
    box-shadow: 0 0 8px var(--ok);
    animation: pulse 2s infinite;
  }
  .pulse-dot.offline {
    background: var(--bad);
    box-shadow: 0 0 8px var(--bad);
    animation: none;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
  }

  /* Control Actions in Header */
  .control-group {
    display: flex;
    gap: 6px;
  }
  .btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    font-family: var(--font-mono);
    cursor: pointer;
    border: 1px solid var(--border-dim);
    background: var(--bg-subtle);
    color: var(--text-main);
    transition: all 0.15s ease;
  }
  .btn:hover:not(:disabled) {
    background: var(--border-dim);
    border-color: var(--border-highlight);
    transform: translateY(-1px);
  }
  .btn:active:not(:disabled) {
    transform: translateY(0);
  }
  .btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  .btn-start { border-color: rgba(16, 185, 129, 0.4); color: #34d399; }
  .btn-start:hover:not(:disabled) { background: rgba(16, 185, 129, 0.15); border-color: var(--ok); }
  .btn-stop { border-color: rgba(239, 68, 68, 0.4); color: #f87171; }
  .btn-stop:hover:not(:disabled) { background: rgba(239, 68, 68, 0.15); border-color: var(--bad); }
  .btn-restart { border-color: rgba(56, 189, 248, 0.4); color: #7dd3fc; }
  .btn-restart:hover:not(:disabled) { background: rgba(56, 189, 248, 0.15); border-color: var(--accent); }

  /* Main Container */
  .container {
    max-width: 1440px;
    margin: 20px auto;
    padding: 0 24px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  /* Top Stat Strip Grid */
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
  }
  @media (max-width: 1024px) {
    .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  }
  @media (max-width: 640px) {
    .kpi-grid { grid-template-columns: 1fr; }
  }

  .kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border-dim);
    border-radius: 8px;
    padding: 14px 16px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
  }
  .kpi-card:hover {
    border-color: var(--border-line);
  }
  .kpi-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }
  .kpi-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
  }
  .kpi-badge {
    font-size: 10px;
    font-family: var(--font-mono);
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 600;
  }
  .badge-ok { background: var(--ok-glow); color: var(--ok); border: 1px solid rgba(16,185,129,0.3); }
  .badge-warn { background: var(--warn-glow); color: var(--warn); border: 1px solid rgba(245,158,11,0.3); }
  .badge-bad { background: var(--bad-glow); color: var(--bad); border: 1px solid rgba(239,68,68,0.3); }
  .badge-info { background: var(--accent-glow); color: var(--accent); border: 1px solid rgba(56,189,248,0.3); }

  .kpi-main-val {
    font-size: 22px;
    font-weight: 700;
    font-family: var(--font-mono);
    letter-spacing: -0.02em;
    color: var(--text-main);
    margin-bottom: 6px;
  }
  .kpi-sub {
    font-size: 11px;
    color: var(--text-secondary);
    display: flex;
    justify-content: space-between;
    font-family: var(--font-mono);
  }

  /* Progress Bar in KPI */
  .bar-container {
    height: 4px;
    background: var(--bg-subtle);
    border-radius: 2px;
    margin: 8px 0;
    overflow: hidden;
    position: relative;
  }
  .bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--ok));
    border-radius: 2px;
    transition: width 0.4s ease;
  }

  /* 2-Column Section Layout */
  .layout-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }
  @media (max-width: 1024px) {
    .layout-grid { grid-template-columns: 1fr; }
  }

  .panel {
    background: var(--bg-surface);
    border: 1px solid var(--border-dim);
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .panel-header {
    padding: 12px 18px;
    background: var(--bg-card);
    border-bottom: 1px solid var(--border-dim);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .panel-title {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .panel-title i {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
  }
  .panel-body {
    padding: 16px 18px;
    flex: 1;
  }

  /* Table Style Rows */
  .dense-table {
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .dense-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 4px 0;
    border-bottom: 1px solid rgba(255,255,255,0.03);
    font-size: 12px;
  }
  .dense-row:last-child { border-bottom: none; }
  .dense-key {
    color: var(--text-muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .dense-val {
    font-family: var(--font-mono);
    color: var(--text-main);
    font-variant-numeric: tabular-nums;
  }

  /* Interactive Timeline Canvas Section */
  .chart-wrapper {
    position: relative;
    margin-top: 8px;
    background: #090c12;
    border: 1px solid var(--border-dim);
    border-radius: 6px;
    padding: 12px 14px 8px;
  }
  .chart-canvas {
    width: 100%;
    height: 180px;
    display: block;
    cursor: crosshair;
  }
  .chart-legend {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 10px;
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--text-muted);
  }
  .legend-items {
    display: flex;
    gap: 16px;
  }
  .legend-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .legend-color {
    width: 10px;
    height: 3px;
    border-radius: 1px;
  }

  /* Tooltip for Chart Scrub */
  .chart-tooltip {
    position: absolute;
    top: 14px;
    right: 14px;
    background: rgba(15, 23, 42, 0.95);
    border: 1px solid var(--border-highlight);
    border-radius: 6px;
    padding: 6px 10px;
    font-family: var(--font-mono);
    font-size: 11px;
    pointer-events: none;
    display: none;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
  }

  /* Speculative Decoding Bar Chart */
  .mtp-bars {
    display: flex;
    gap: 6px;
    align-items: flex-end;
    height: 50px;
    margin-top: 10px;
    padding: 4px;
    background: var(--bg-subtle);
    border-radius: 6px;
    border: 1px solid var(--border-dim);
  }
  .mtp-col {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    height: 100%;
    justify-content: flex-end;
    gap: 2px;
  }
  .mtp-bar {
    width: 100%;
    background: var(--accent);
    border-radius: 2px 2px 0 0;
    transition: height 0.3s ease;
    min-height: 2px;
  }
  .mtp-label {
    font-size: 9px;
    font-family: var(--font-mono);
    color: var(--text-muted);
  }

  /* Capacity Ledger Box */
  .capacity-box {
    margin-top: 12px;
    background: #090c14;
    border: 1px solid var(--border-dim);
    border-radius: 6px;
    padding: 10px 12px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-secondary);
    line-height: 1.6;
    word-break: break-all;
  }
  .capacity-highlight {
    color: var(--accent);
    font-weight: 600;
  }

  /* Terminal Log Section */
  .log-container {
    display: flex;
    flex-direction: column;
    height: 380px;
    background: #06080d;
    border: 1px solid var(--border-dim);
    border-radius: 6px;
    overflow: hidden;
  }
  .log-toolbar {
    padding: 8px 12px;
    background: var(--bg-card);
    border-bottom: 1px solid var(--border-dim);
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
  }
  .log-filters {
    display: flex;
    gap: 6px;
  }
  .log-filter-btn {
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-family: var(--font-mono);
    border: 1px solid var(--border-dim);
    background: var(--bg-subtle);
    color: var(--text-secondary);
    cursor: pointer;
  }
  .log-filter-btn.active {
    background: var(--border-highlight);
    color: var(--text-main);
    border-color: var(--accent);
  }
  .log-search {
    background: var(--bg-input);
    border: 1px solid var(--border-dim);
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--text-main);
    width: 140px;
  }
  .log-search:focus {
    outline: none;
    border-color: var(--accent);
  }
  .log-scroll-pane {
    flex: 1;
    overflow-y: auto;
    padding: 10px 14px;
    font-family: var(--font-mono);
    font-size: 11.5px;
    line-height: 1.45;
    color: #cbd5e1;
    white-space: pre-wrap;
    word-break: break-all;
  }
  .log-line-info { color: #38bdf8; }
  .log-line-warn { color: #fbbf24; font-weight: 600; }
  .log-line-error { color: #f87171; font-weight: 600; }
  .log-line-req { color: #34d399; }
  .log-line-dim { color: var(--text-dim); }

  /* Insights Section */
  .insights-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 10px;
  }
  .insight-card {
    background: var(--bg-card);
    border: 1px solid var(--border-dim);
    border-left: 3px solid var(--border-highlight);
    border-radius: 6px;
    padding: 12px 16px;
    transition: border-color 0.2s;
  }
  .insight-card.critical { border-left-color: var(--bad); }
  .insight-card.warning { border-left-color: var(--warn); }
  .insight-card.info { border-left-color: var(--accent); }
  .insight-card.available { border-left-color: var(--ok); }
  .insight-card:hover { border-color: var(--border-line); }

  .insight-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
  }
  .insight-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-main);
  }
  .insight-stmt {
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 8px;
    line-height: 1.4;
  }
  .insight-rec {
    background: rgba(56, 189, 248, 0.08);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 11px;
    color: #bae6fd;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .insight-rec::before {
    content: "💡";
    font-size: 12px;
  }
  .insight-details {
    margin-top: 8px;
  }
  .insight-details summary {
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--text-muted);
    cursor: pointer;
    user-select: none;
  }
  .insight-details summary:hover { color: var(--text-secondary); }
  .insight-evidence {
    margin-top: 6px;
    background: var(--bg-base);
    border: 1px solid var(--border-dim);
    border-radius: 4px;
    padding: 8px 10px;
    font-family: var(--font-mono);
    font-size: 10.5px;
    max-height: 140px;
    overflow-y: auto;
    color: #94a3b8;
  }

  /* Full Width Span */
  .span-full { grid-column: 1 / -1; }
</style>
</head>
<body>

<header>
  <div class="brand">
    <div class="brand-logo">N</div>
    <div>
      <div class="brand-title">NInfer Supervisor</div>
      <div class="brand-sub">PRECISION MISSION CONTROL // PORT 8099</div>
    </div>
  </div>

  <div class="header-badges">
    <div class="pulse-badge">
      <div id="live-dot" class="pulse-dot"></div>
      <span id="live-label">STREAMING LIVE</span>
    </div>

    <div id="controls" class="control-group">
      <button class="btn btn-start" data-act="start" id="btn-start">▶ Start</button>
      <button class="btn btn-stop" data-act="stop" id="btn-stop">■ Stop</button>
      <button class="btn btn-restart" data-act="restart" id="btn-restart">⟳ Restart</button>
    </div>
  </div>
</header>

<div class="container">

  <!-- 4 Key Performance Metric Tiles -->
  <div class="kpi-grid">
    <!-- Tile 1: Engine State -->
    <div class="kpi-card">
      <div class="kpi-header">
        <span class="kpi-title">Engine State</span>
        <span id="kpi-state-badge" class="kpi-badge badge-warn">INITIALIZING</span>
      </div>
      <div id="kpi-state-main" class="kpi-main-val">—</div>
      <div class="kpi-sub">
        <span>PID: <strong id="kpi-pid" style="color:var(--text-main)">—</strong></span>
        <span>UPTIME: <strong id="kpi-uptime" style="color:var(--text-main)">—</strong></span>
      </div>
    </div>

    <!-- Tile 2: VRAM Allocation -->
    <div class="kpi-card">
      <div class="kpi-header">
        <span class="kpi-title">VRAM Allocation</span>
        <span id="kpi-vram-pct" class="kpi-badge badge-info">0%</span>
      </div>
      <div id="kpi-vram-main" class="kpi-main-val">—</div>
      <div class="bar-container">
        <div id="kpi-vram-bar" class="bar-fill" style="width: 0%"></div>
      </div>
      <div class="kpi-sub">
        <span>DXGI BUDGET: <strong id="kpi-dxgi-budget" style="color:var(--accent)">—</strong></span>
        <span>TOTAL: <strong id="kpi-vram-total" style="color:var(--text-secondary)">—</strong></span>
      </div>
    </div>

    <!-- Tile 3: Inference Rates -->
    <div class="kpi-card">
      <div class="kpi-header">
        <span class="kpi-title">Inference Throughput</span>
        <span class="kpi-badge badge-ok">PORT :8010</span>
      </div>
      <div id="kpi-decode-rate" class="kpi-main-val">— tok/s</div>
      <div class="kpi-sub">
        <span>MEAN TTFT: <strong id="kpi-ttft" style="color:var(--ok)">—</strong></span>
        <span>DONE REQS: <strong id="kpi-done" style="color:var(--text-main)">—</strong></span>
      </div>
    </div>

    <!-- Tile 4: MTP Speculative Decoding -->
    <div class="kpi-card">
      <div class="kpi-header">
        <span class="kpi-title">MTP Speculative Rate</span>
        <span id="kpi-mtp-badge" class="kpi-badge badge-info">MTP5</span>
      </div>
      <div id="kpi-mtp-pct" class="kpi-main-val">—</div>
      <div class="kpi-sub">
        <span>ACCEPTED: <strong id="kpi-mtp-ratio" style="color:var(--accent)">—</strong></span>
        <span>FALLBACKS: <strong id="kpi-mtp-fallbacks" style="color:var(--warn)">—</strong></span>
      </div>
    </div>
  </div>

  <!-- Interactive Timeline Chart Full-Width Panel -->
  <div class="panel">
    <div class="panel-header">
      <div class="panel-title">
        <i></i> VRAM Allocation &amp; WDDM System Pressure (10 Hz High-Density Stream)
      </div>
      <div class="legend-items">
        <span class="legend-tag"><i class="legend-color" style="background:var(--ok)"></i> Physical VRAM (NVIDIA-SMI)</span>
        <span class="legend-tag"><i class="legend-color" style="background:var(--accent)"></i> WDDM Pressure Budget (DXGI)</span>
        <span class="legend-tag"><i class="legend-color" style="background:var(--warn);width:3px;height:10px"></i> Engine / Admin Event</span>
      </div>
    </div>
    <div class="panel-body" style="padding-bottom:12px;">
      <div class="chart-wrapper">
        <canvas id="timeline-canvas" class="chart-canvas"></canvas>
        <div id="chart-tooltip" class="chart-tooltip"></div>
      </div>
    </div>
  </div>

  <!-- Main 2-Column Grid -->
  <div class="layout-grid">

    <!-- Column Left: Detailed Telemetry & Speculative Breakdown -->
    <div style="display:flex; flex-direction:column; gap:20px;">

      <!-- GPU & KV Sizing Details -->
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title"><i></i> Hardware &amp; KV Cache Architecture</div>
          <span id="adapter-tag" class="kpi-badge badge-info">—</span>
        </div>
        <div class="panel-body">
          <div class="dense-table">
            <div class="dense-row">
              <span class="dense-key">GPU Device</span>
              <span id="dt-gpu-name" class="dense-val">—</span>
            </div>
            <div class="dense-row">
              <span class="dense-key">Physical Memory</span>
              <span id="dt-phys-vram" class="dense-val">—</span>
            </div>
            <div class="dense-row">
              <span class="dense-key">DXGI System Budget</span>
              <span id="dt-dxgi-budget" class="dense-val">—</span>
            </div>
            <div class="dense-row">
              <span class="dense-key">Supervisor Memory Overhead</span>
              <span id="dt-sup-vram" class="dense-val">—</span>
            </div>
            <div class="dense-row">
              <span class="dense-key">Last Lifecycle Event</span>
              <span id="dt-last-event" class="dense-val" style="color:var(--accent)">—</span>
            </div>
            <div class="dense-row">
              <span class="dense-key">Health Status Check</span>
              <span id="dt-health-check" class="dense-val">—</span>
            </div>
          </div>

          <div class="capacity-box">
            <div style="color:var(--text-muted); font-size:10px; margin-bottom:4px; text-transform:uppercase;">Boot Capacity Resolution Line</div>
            <div id="dt-capacity-text">Waiting for engine capacity resolution...</div>
          </div>
        </div>
      </div>

      <!-- Speculative Acceptance Breakdown -->
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title"><i></i> MTP Multi-Token Speculative Acceptance</div>
          <span id="dt-mtp-rate-badge" class="kpi-badge badge-ok">—</span>
        </div>
        <div class="panel-body">
          <div class="dense-table">
            <div class="dense-row">
              <span class="dense-key">Speculative Backend</span>
              <span id="dt-mtp-backend" class="dense-val">MTP (Multi-Token Prediction)</span>
            </div>
            <div class="dense-row">
              <span class="dense-key">Draft Window</span>
              <span id="dt-mtp-window" class="dense-val">5 Draft Tokens</span>
            </div>
            <div class="dense-row">
              <span class="dense-key">Prefix Cache Reuse Mix</span>
              <span id="dt-reuse-mix" class="dense-val">—</span>
            </div>
          </div>

          <div style="margin-top:14px;">
            <div style="display:flex; justify-content:space-between; font-size:10px; font-family:var(--font-mono); color:var(--text-muted); text-transform:uppercase; margin-bottom:4px;">
              <span>Acceptance Per Speculative Position [1..5]</span>
              <span id="mtp-pos-legend">Position Ratios</span>
            </div>
            <div id="mtp-bars-container" class="mtp-bars">
              <div class="mtp-col"><div class="mtp-bar" style="height:0%"></div><span class="mtp-label">P1</span></div>
              <div class="mtp-col"><div class="mtp-bar" style="height:0%"></div><span class="mtp-label">P2</span></div>
              <div class="mtp-col"><div class="mtp-bar" style="height:0%"></div><span class="mtp-label">P3</span></div>
              <div class="mtp-col"><div class="mtp-bar" style="height:0%"></div><span class="mtp-label">P4</span></div>
              <div class="mtp-col"><div class="mtp-bar" style="height:0%"></div><span class="mtp-label">P5</span></div>
            </div>
          </div>
        </div>
      </div>

    </div>

    <!-- Column Right: Real-time Terminal Log & Insights -->
    <div style="display:flex; flex-direction:column; gap:20px;">

      <!-- Streaming Engine Console -->
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title"><i></i> Live Engine Log Stream</div>
          <div style="display:flex; gap:8px; align-items:center;">
            <button id="btn-autoscroll" class="log-filter-btn active" title="Toggle Autoscroll Lock">🔒 Auto-Scroll</button>
            <button id="btn-copylog" class="log-filter-btn" title="Copy Log to Clipboard">📋 Copy</button>
          </div>
        </div>
        <div class="log-container">
          <div class="log-toolbar">
            <div class="log-filters">
              <button class="log-filter-btn active" data-filter="all">All</button>
              <button class="log-filter-btn" data-filter="req">Requests</button>
              <button class="log-filter-btn" data-filter="throughput">Throughput</button>
              <button class="log-filter-btn" data-filter="warn">Alerts</button>
            </div>
            <input type="text" id="log-search" class="log-search" placeholder="Filter log..." />
          </div>
          <div id="log-content" class="log-scroll-pane">Connecting to engine log tail...</div>
        </div>
      </div>

      <!-- Autonomous Diagnostics & Insights -->
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title"><i></i> Performance &amp; Cache Insights</div>
          <span id="insights-count-badge" class="kpi-badge badge-info">0 INSIGHTS</span>
        </div>
        <div class="panel-body">
          <div id="insights-container" class="insights-grid">
            <div style="color:var(--text-muted); font-size:12px; font-family:var(--font-mono); text-align:center; padding:16px;">
              Analyzing request logs for prefix collapse, latency saturation, and cache pressure...
            </div>
          </div>
        </div>
      </div>

    </div>

  </div>

</div>

<script>
(function() {
  'use strict';

  // Formatting Helpers
  function gib(bytes) {
    if (bytes == null || isNaN(bytes)) return '—';
    return (bytes / 1073741824).toFixed(2) + ' GiB';
  }
  function mib(bytes) {
    if (bytes == null || isNaN(bytes)) return '—';
    return (bytes / 1048576).toFixed(1) + ' MiB';
  }
  function formatUptime(seconds) {
    if (seconds == null || isNaN(seconds)) return '—';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}h ${m}m ${s}s`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
  }
  function esc(str) {
    return String(str == null ? '' : str).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
  }

  // State Management
  let lastState = null;
  let autoScroll = true;
  let activeLogFilter = 'all';
  let logSearchQuery = '';
  let rawLogTail = '';

  // UI Element Refs
  const liveDot = document.getElementById('live-dot');
  const liveLabel = document.getElementById('live-label');
  const btnStart = document.getElementById('btn-start');
  const btnStop = document.getElementById('btn-stop');
  const btnRestart = document.getElementById('btn-restart');
  const btnAutoscroll = document.getElementById('btn-autoscroll');
  const btnCopyLog = document.getElementById('btn-copylog');
  const logContent = document.getElementById('log-content');
  const logSearch = document.getElementById('log-search');
  const canvas = document.getElementById('timeline-canvas');
  const tooltip = document.getElementById('chart-tooltip');

  // Canvas High-DPI Setup
  const ctx = canvas.getContext('2d');
  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    if (lastState && lastState.series) {
      drawTimeline(lastState.series);
    }
  }
  window.addEventListener('resize', resizeCanvas);
  setTimeout(resizeCanvas, 50);

  // Draw High-Density 60fps Canvas Timeline
  let mousePos = null;
  canvas.addEventListener('mousemove', e => {
    const rect = canvas.getBoundingClientRect();
    mousePos = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    if (lastState && lastState.series) drawTimeline(lastState.series);
  });
  canvas.addEventListener('mouseleave', () => {
    mousePos = null;
    tooltip.style.display = 'none';
    if (lastState && lastState.series) drawTimeline(lastState.series);
  });

  function drawTimeline(ser) {
    if (!ser) return;
    const b = ser.budget_bytes || [];
    const u = ser.nvidia_used_bytes || [];
    const t = ser.t_ms || [];
    const n = b.length;
    const rect = canvas.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;

    ctx.clearRect(0, 0, W, H);
    if (n < 2) return;

    let yMax = 1;
    for (let i = 0; i < n; i++) {
      if (b[i] > yMax) yMax = b[i];
      if (u[i] > yMax) yMax = u[i];
    }
    yMax = yMax * 1.08; // 8% headroom

    const pX = 40;
    const pY = 16;
    const pBottom = 22;
    const graphW = W - pX - 16;
    const graphH = H - pY - pBottom;

    // Draw Subtle Horizontal Grid Lines
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#1e293b';
    ctx.fillStyle = '#64748b';
    ctx.font = '10px "Cascadia Code", monospace';
    ctx.textAlign = 'right';

    const steps = 4;
    for (let s = 0; s <= steps; s++) {
      const val = (yMax / steps) * s;
      const y = pY + graphH - (graphH * (s / steps));
      ctx.beginPath();
      ctx.moveTo(pX, y);
      ctx.lineTo(W - 16, y);
      ctx.stroke();
      ctx.fillText((val / 1073741824).toFixed(0) + 'G', pX - 6, y + 3);
    }

    const t0 = t[0] || 0;
    const t1 = t[n - 1] || t0;
    const span = Math.max(t1 - t0, 1);

    const getX = i => pX + graphW * (i / (n - 1));
    const getY = val => pY + graphH - (graphH * (Math.max(0, val || 0) / yMax));

    // Fill Area for Physical VRAM (Emerald)
    const physGrad = ctx.createLinearGradient(0, pY, 0, pY + graphH);
    physGrad.addColorStop(0, 'rgba(16, 185, 129, 0.25)');
    physGrad.addColorStop(1, 'rgba(16, 185, 129, 0.00)');

    ctx.beginPath();
    ctx.moveTo(getX(0), pY + graphH);
    for (let i = 0; i < n; i++) {
      ctx.lineTo(getX(i), getY(u[i]));
    }
    ctx.lineTo(getX(n - 1), pY + graphH);
    ctx.closePath();
    ctx.fillStyle = physGrad;
    ctx.fill();

    // Stroke Physical VRAM
    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#10b981';
    for (let i = 0; i < n; i++) {
      if (i === 0) ctx.moveTo(getX(i), getY(u[i]));
      else ctx.lineTo(getX(i), getY(u[i]));
    }
    ctx.stroke();

    // Stroke DXGI System Pressure Budget (Cyan/Sky)
    ctx.beginPath();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = '#38bdf8';
    for (let i = 0; i < n; i++) {
      if (i === 0) ctx.moveTo(getX(i), getY(b[i]));
      else ctx.lineTo(getX(i), getY(b[i]));
    }
    ctx.stroke();

    // Event Markers (Vertical Flags)
    (ser.events || []).forEach(ev => {
      const evX = pX + graphW * ((ev.t_ms - t0) / span);
      if (evX >= pX && evX <= W - 16) {
        ctx.beginPath();
        ctx.strokeStyle = ev.kind && ev.kind.includes('boot') ? '#38bdf8' : '#f59e0b';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.moveTo(evX, pY);
        ctx.lineTo(evX, pY + graphH);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    });

    // Crosshair Hover Scrub
    if (mousePos && mousePos.x >= pX && mousePos.x <= W - 16) {
      const ratio = (mousePos.x - pX) / graphW;
      const idx = Math.min(n - 1, Math.max(0, Math.round(ratio * (n - 1))));
      const curX = getX(idx);

      ctx.beginPath();
      ctx.strokeStyle = '#94a3b8';
      ctx.lineWidth = 1;
      ctx.moveTo(curX, pY);
      ctx.lineTo(curX, pY + graphH);
      ctx.stroke();

      // Highlight Points
      ctx.fillStyle = '#10b981';
      ctx.beginPath();
      ctx.arc(curX, getY(u[idx]), 4, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#38bdf8';
      ctx.beginPath();
      ctx.arc(curX, getY(b[idx]), 4, 0, Math.PI * 2);
      ctx.fill();

      // Tooltip HUD
      tooltip.style.display = 'block';
      tooltip.innerHTML = `
        <div style="color:#f1f5f9; font-weight:700; margin-bottom:2px;">T - ${((t1 - t[idx])/1000).toFixed(1)}s</div>
        <div style="color:#10b981">VRAM: ${gib(u[idx])}</div>
        <div style="color:#38bdf8">DXGI: ${gib(b[idx])}</div>
      `;
    }
  }

  // Format & Syntax Highlight Engine Log Lines
  function formatLogText(rawText) {
    if (!rawText) return 'Log is empty or waiting for engine lines...';
    const lines = rawText.split('\n');
    const filtered = lines.filter(l => {
      if (!l.trim()) return false;
      if (logSearchQuery && !l.toLowerCase().includes(logSearchQuery.toLowerCase())) return false;
      if (activeLogFilter === 'req') return l.includes('[req ');
      if (activeLogFilter === 'throughput') return l.includes('throughput interval');
      if (activeLogFilter === 'warn') return l.includes('[warn]') || l.includes('[error]') || l.includes('fail');
      return true;
    });

    return filtered.map(l => {
      let cls = 'log-line-dim';
      if (l.includes('[error]') || l.includes('failed') || l.includes('HALT')) cls = 'log-line-error';
      else if (l.includes('[warn]') || l.includes('WARNING')) cls = 'log-line-warn';
      else if (l.includes('[req ')) cls = 'log-line-req';
      else if (l.includes('[info]')) cls = 'log-line-info';

      return `<div class="${cls}">${esc(l)}</div>`;
    }).join('');
  }

  // Update Application State
  function render(s) {
    if (!s) return;
    lastState = s;

    // Heartbeat
    liveDot.className = 'pulse-dot';
    liveLabel.textContent = s.monitor_only ? 'MONITOR ONLY (UNMANAGED)' : 'STREAMING LIVE (SSE)';

    const eng = s.engine || {};
    const state = eng.state || 'Unknown';

    // State Badge in Header & KPI
    const kpiBadge = document.getElementById('kpi-state-badge');
    const kpiStateMain = document.getElementById('kpi-state-main');
    kpiStateMain.textContent = state.toUpperCase();

    if (state === 'Running') {
      kpiBadge.className = 'kpi-badge badge-ok';
      kpiBadge.textContent = 'ONLINE';
    } else if (state === 'Halted') {
      kpiBadge.className = 'kpi-badge badge-bad';
      kpiBadge.textContent = 'HALTED';
    } else {
      kpiBadge.className = 'kpi-badge badge-warn';
      kpiBadge.textContent = state.toUpperCase();
    }

    document.getElementById('kpi-pid').textContent = eng.pid || '—';
    document.getElementById('kpi-uptime').textContent = formatUptime(eng.uptime_s);

    // VRAM Metrics
    const nv = s.nvidia_smi || {};
    const dxgi = s.dxgi || {};
    const usedBytes = nv.used_bytes || 0;
    const totalBytes = nv.total_bytes || (102641500160); // 95.6 GB fallback
    const pct = totalBytes ? ((usedBytes / totalBytes) * 100).toFixed(1) : 0;

    document.getElementById('kpi-vram-main').textContent = gib(usedBytes);
    document.getElementById('kpi-vram-pct').textContent = pct + '%';
    document.getElementById('kpi-vram-bar').style.width = pct + '%';
    document.getElementById('kpi-dxgi-budget').textContent = gib(dxgi.budget_bytes);
    document.getElementById('kpi-vram-total').textContent = gib(totalBytes);

    // Hardware Details
    document.getElementById('adapter-tag').textContent = dxgi.adapter_name ? 'RTX PRO 6000' : 'GPU 0';
    document.getElementById('dt-gpu-name').textContent = dxgi.adapter_name || 'NVIDIA GPU';
    document.getElementById('dt-phys-vram').textContent = `${gib(usedBytes)} / ${gib(totalBytes)} (${pct}%)`;
    document.getElementById('dt-dxgi-budget').textContent = gib(dxgi.budget_bytes);
    document.getElementById('dt-sup-vram').textContent = mib(dxgi.supervisor_usage_bytes);
    document.getElementById('dt-last-event').textContent = eng.last_event || '—';
    document.getElementById('dt-health-check').textContent = s.health ? `HTTP ${s.health.status} (${s.health.body || 'OK'})` : '—';

    if (s.engine_capacity_line) {
      document.getElementById('dt-capacity-text').textContent = s.engine_capacity_line;
    }

    // Inference & Throughput Metrics
    const reqs = s.requests || {};
    document.getElementById('kpi-decode-rate').textContent = reqs.decode_tok_s_mean ? `${reqs.decode_tok_s_mean.toFixed(1)} tok/s` : '— tok/s';
    document.getElementById('kpi-ttft').textContent = reqs.ttft_ms_mean ? `${reqs.ttft_ms_mean.toFixed(0)} ms` : '—';
    document.getElementById('kpi-done').textContent = reqs.done != null ? reqs.done.toLocaleString() : '—';

    // MTP Speculative Telemetry
    if (reqs.mtp_drafted) {
      const ratePct = ((reqs.mtp_last_accept_rate || 0) * 100).toFixed(1);
      document.getElementById('kpi-mtp-pct').textContent = ratePct + '%';
      document.getElementById('kpi-mtp-ratio').textContent = `${(reqs.mtp_accepted||0).toLocaleString()} / ${(reqs.mtp_drafted||0).toLocaleString()}`;
      document.getElementById('kpi-mtp-fallbacks').textContent = reqs.mtp_fallback_steps || 0;
      document.getElementById('dt-mtp-rate-badge').textContent = `${ratePct}% ACCEPTED`;

      // Render 5-Position Speculative Bars
      const pos = reqs.mtp_accepted_per_position || [];
      const maxPos = Math.max(1, ...pos);
      const cols = document.querySelectorAll('#mtp-bars-container .mtp-bar');
      pos.forEach((pVal, idx) => {
        if (cols[idx]) {
          const barHeight = Math.max(4, Math.round((pVal / maxPos) * 100));
          cols[idx].style.height = barHeight + '%';
          cols[idx].title = `Pos ${idx+1}: ${pVal} accepted (${((pVal/(reqs.mtp_accepted||1))*100).toFixed(0)}%)`;
        }
      });
    }

    // Reuse Mix
    const reuseStr = `Seed: ${reqs.reuse_seed || 0} · Append: ${reqs.reuse_append || 0} · Reset: ${reqs.reuse_full_reset || 0}`;
    document.getElementById('dt-reuse-mix').textContent = reuseStr;

    // Log Stream
    rawLogTail = s.log_tail || '';
    logContent.innerHTML = formatLogText(rawLogTail);
    if (autoScroll) {
      logContent.scrollTop = logContent.scrollHeight;
    }

    // Diagnostics & Insights Cards
    const rep = s.insights || {};
    const items = rep.insights || [];
    document.getElementById('insights-count-badge').textContent = `${items.length} INSIGHTS`;

    const insContainer = document.getElementById('insights-container');
    if (items.length === 0) {
      insContainer.innerHTML = `<div style="color:var(--text-muted); font-size:12px; font-family:var(--font-mono); text-align:center; padding:16px;">All operational metrics within optimal boundaries. No anomalies detected.</div>`;
    } else {
      insContainer.innerHTML = items.map((it, idx) => {
        const sev = it.severity || 'info';
        const avail = it.availability || 'available';
        return `
          <div class="insight-card ${sev}">
            <div class="insight-top">
              <span class="insight-title">${esc(it.title || it.id)}</span>
              <span class="kpi-badge badge-${sev === 'critical' ? 'bad' : sev === 'warning' ? 'warn' : 'info'}">${esc(sev.toUpperCase())}</span>
            </div>
            <div class="insight-stmt">${esc(it.statement || '')}</div>
            ${it.recommendation ? `<div class="insight-rec">${esc(it.recommendation)}</div>` : ''}
            <details class="insight-details">
              <summary>View Diagnostic Telemetry &amp; Evidence</summary>
              <pre class="insight-evidence">${esc(JSON.stringify(it.evidence || {}, null, 2))}</pre>
            </details>
          </div>
        `;
      }).join('');
    }

    // Timeline Rendering
    drawTimeline(s.series);
  }

  // Control Actions
  async function dispatchAction(actionName, btn) {
    if (!confirm(`Confirm ${actionName.toUpperCase()} action on NInfer engine?`)) return;
    const oldText = btn.textContent;
    btn.disabled = true;
    btn.textContent = '...';
    try {
      const resp = await fetch(`/api/${actionName}`, {
        method: 'POST',
        headers: { 'X-NInfer-Supervisor': '1' }
      });
      const data = await resp.json();
      render(data);
    } catch (err) {
      alert(`Action ${actionName} failed: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = oldText;
    }
  }

  btnStart.onclick = () => dispatchAction('start', btnStart);
  btnStop.onclick = () => dispatchAction('stop', btnStop);
  btnRestart.onclick = () => dispatchAction('restart', btnRestart);

  // Log Controls
  btnAutoscroll.onclick = () => {
    autoScroll = !autoScroll;
    btnAutoscroll.classList.toggle('active', autoScroll);
    btnAutoscroll.textContent = autoScroll ? '🔒 Auto-Scroll' : '🔓 Scrolled';
  };

  btnCopyLog.onclick = () => {
    navigator.clipboard.writeText(rawLogTail).then(() => {
      const orig = btnCopyLog.textContent;
      btnCopyLog.textContent = '✓ Copied!';
      setTimeout(() => { btnCopyLog.textContent = orig; }, 1500);
    });
  };

  document.querySelectorAll('.log-filter-btn[data-filter]').forEach(b => {
    b.onclick = () => {
      document.querySelectorAll('.log-filter-btn[data-filter]').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      activeLogFilter = b.dataset.filter;
      logContent.innerHTML = formatLogText(rawLogTail);
    };
  });

  logSearch.oninput = e => {
    logSearchQuery = e.target.value;
    logContent.innerHTML = formatLogText(rawLogTail);
  };

  // SSE & Initial Fetch
  const es = new EventSource('/api/events');
  es.onmessage = e => {
    try { render(JSON.parse(e.data)); } catch (err) {}
  };
  es.onerror = () => {
    liveDot.className = 'pulse-dot offline';
    liveLabel.textContent = 'DISCONNECTED (RECONNECTING)';
  };

  fetch('/api/state').then(r => r.json()).then(render).catch(() => {});
})();
</script>
</body>
</html>
)HTML";

} // namespace ninfer::supervisor
