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
    --text-main: #f8fafc;
    --text-secondary: #cbd5e1;
    --text-muted: #94a3b8;
    --text-dim: #8092a8;
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

  :focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }

  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border-width: 0;
  }

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
    margin: 0;
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
    will-change: transform, opacity;
  }
  .pulse-dot.offline {
    background: var(--bad);
    box-shadow: 0 0 8px var(--bad);
    animation: none;
  }
  .pulse-dot.paused {
    background: var(--warn);
    box-shadow: 0 0 8px var(--warn);
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

  /* Info Help Tip Icon */
  .help-tip {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--bg-subtle);
    border: 1px solid var(--border-highlight);
    color: var(--text-muted);
    font-size: 9px;
    font-family: var(--font-mono);
    font-weight: 700;
    cursor: help;
    margin-left: 4px;
    user-select: none;
    transition: color 0.15s, border-color 0.15s;
  }
  .help-tip:hover {
    color: var(--accent);
    border-color: var(--accent);
  }

  /* Main Container */
  main.container {
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
    display: flex;
    align-items: center;
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

  /* Progress Bar in KPI with GPU transform */
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
    width: 100%;
    background: linear-gradient(90deg, var(--accent), var(--ok));
    border-radius: 2px;
    transform-origin: left center;
    transform: scaleX(0);
    transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    will-change: transform;
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
  h2.panel-title {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0;
  }
  h2.panel-title i {
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
    display: inline-flex;
    align-items: center;
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
  .chart-canvas:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 1px;
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

  /* Speculative Decoding Bar Chart with GPU transform */
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
  .mtp-bar-track {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: flex-end;
  }
  .mtp-bar {
    width: 100%;
    height: 100%;
    background: var(--accent);
    border-radius: 2px 2px 0 0;
    transform-origin: center bottom;
    transform: scaleY(0);
    transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    will-change: transform;
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
)HTML"
R"HTML(  .log-scroll-pane {
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
    border-radius: 6px;
    padding: 12px 16px;
    transition: border-color 0.2s;
  }
  .insight-card.critical { border-top: 2px solid var(--bad); }
  .insight-card.warning { border-top: 2px solid var(--warn); }
  .insight-card.info { border-top: 2px solid var(--accent); }
  .insight-card.available { border-top: 2px solid var(--ok); }
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
    color: var(--text-muted);
  }

  /* In-Palette Non-Blocking Modal */
  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(4, 6, 10, 0.75);
    backdrop-filter: blur(8px);
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .modal-card {
    background: var(--bg-card);
    border: 1px solid var(--border-highlight);
    border-radius: 8px;
    padding: 20px 24px;
    max-width: 420px;
    width: 90%;
    box-shadow: 0 16px 32px rgba(0,0,0,0.6);
  }
  .modal-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--text-main);
    margin-bottom: 8px;
  }
  .modal-desc {
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 20px;
    line-height: 1.5;
  }
  .modal-actions {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
  }
  /* Configuration editor */
  .cfg-tabs { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:12px; }
  .cfg-groups { display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:10px 20px; }
  .cfg-field { display:flex; flex-direction:column; gap:4px; padding:8px 0; border-bottom:1px solid var(--border); }
  .cfg-field-top { display:flex; align-items:center; justify-content:space-between; gap:10px; }
  .cfg-label { font-size:11px; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.04em; }
  .cfg-flag { font-family:var(--font-mono); font-size:10px; color:var(--text-dim); }
  .cfg-help { font-size:11px; color:var(--text-muted); line-height:1.45; }
  .cfg-input, .cfg-select { background:var(--bg-subtle); border:1px solid var(--border-highlight); color:var(--text-main);
    border-radius:4px; padding:6px 8px; font-family:var(--font-mono); font-size:12px; width:100%; }
  .cfg-input:focus, .cfg-select:focus { outline:2px solid var(--accent); outline-offset:1px; }
  .cfg-input.dirty, .cfg-select.dirty { border-color:var(--warn); }
  .cfg-input[readonly] { color:var(--text-muted); border-style:dashed; }
  .cfg-check { display:flex; align-items:center; gap:8px; font-size:12px; color:var(--text-main); }
  .cfg-actions { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-top:14px;
    padding-top:12px; border-top:1px solid var(--border); flex-wrap:wrap; }
  .cfg-status { font-size:11px; font-family:var(--font-mono); color:var(--text-muted); }
  .cfg-status.dirty { color:var(--warn); }
  .cfg-status.saved { color:var(--ok); }
  .cfg-errors { margin-top:12px; padding:10px 12px; border:1px solid var(--bad); border-left-width:3px;
    border-radius:4px; background:rgba(197,59,51,0.08); font-size:12px; color:var(--text-main); }
  .cfg-errors ul { margin:6px 0 0 16px; padding:0; }
  .cfg-note { grid-column:1/-1; font-size:11px; color:var(--text-muted); line-height:1.5;
    background:var(--bg-subtle); border-left:2px solid var(--border-highlight); padding:8px 10px; border-radius:3px; }
</style>
</head>
<body>

<div id="a11y-live-region" class="sr-only" aria-live="polite" aria-atomic="true"></div>

<!-- Accessible In-Palette Confirmation Modal with Focus Trap -->
<div id="confirm-modal" class="modal-overlay" style="display:none;" role="dialog" aria-modal="true" aria-labelledby="modal-title">
  <div class="modal-card">
    <h2 id="modal-title" class="modal-title">Confirm Engine Action</h2>
    <p id="modal-desc" class="modal-desc">Are you sure you want to proceed with this operation?</p>
    <div class="modal-actions">
      <button id="modal-cancel-btn" class="btn">Cancel (Esc)</button>
      <button id="modal-confirm-btn" class="btn btn-restart">Confirm (Enter)</button>
    </div>
  </div>
</div>

<header>
  <div class="brand">
    <div class="brand-logo" aria-hidden="true">N</div>
    <div>
      <h1 class="brand-title">NInfer Supervisor</h1>
      <div class="brand-sub" id="brand-sub">PRECISION MISSION CONTROL</div>
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

<main class="container">

  <!-- 4 Key Performance Metric Tiles -->
  <section class="kpi-grid" aria-label="Key Performance Indicators">
    <!-- Tile 1: Engine State -->
    <div class="kpi-card">
      <div class="kpi-header">
        <span class="kpi-title">Engine State <span class="help-tip" title="Current process lifecycle state of resident NInfer C++ binary.">?</span></span>
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
        <span class="kpi-title">Device Memory <span class="help-tip" title="Device-wide used and free, read from NVML: every process on the card, not just the engine. cudaMemGetInfo and the DXGI budget both report an empty card while another process holds tens of gigabytes, which is why neither is shown here.">?</span></span>
        <span id="kpi-vram-pct" class="kpi-badge badge-info">0%</span>
      </div>
      <div id="kpi-vram-main" class="kpi-main-val">—</div>
      <div class="bar-container">
        <div id="kpi-vram-bar" class="bar-fill"></div>
      </div>
      <div class="kpi-sub">
        <span>FREE: <strong id="kpi-vram-free" style="color:var(--accent)">—</strong></span>
        <span>TOTAL: <strong id="kpi-vram-total" style="color:var(--text-secondary)">—</strong></span>
      </div>
    </div>

    <!-- Tile 3: Inference Rates -->
    <div class="kpi-card">
      <div class="kpi-header">
        <span class="kpi-title">Inference Throughput <span class="help-tip" title="Rolling window mean token generation speed and time-to-first-token (TTFT).">?</span></span>
        <span id="kpi-engine-port" class="kpi-badge badge-ok">—</span>
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
        <span class="kpi-title">Speculative Decoding <span class="help-tip" title="Share of drafted tokens the model accepted, as reported by the engine per request. Blank when no speculative backend is configured.">?</span></span>
        <span id="kpi-mtp-badge" class="kpi-badge badge-info">—</span>
      </div>
      <div id="kpi-mtp-pct" class="kpi-main-val">—</div>
      <div class="kpi-sub">
        <span>ACCEPTED: <strong id="kpi-mtp-ratio" style="color:var(--accent)">—</strong></span>
        <span>FALLBACKS: <strong id="kpi-mtp-fallbacks" style="color:var(--warn)">—</strong></span>
      </div>
    </div>
  </section>

  <!-- Interactive Timeline Chart Full-Width Panel -->
  <section class="panel" aria-label="VRAM Timeline and Pressure">
    <div class="panel-header">
      <h2 class="panel-title">
        <i></i> Device Memory Over Time (10 Hz, device-wide from NVIDIA-SMI)
        <span class="help-tip" title="Real-time 10 Hz physical VRAM usage (Emerald) and Windows DXGI budget line (Cyan). Vertical dashed flags denote lifecycle events.">?</span>
      </h2>
      <div class="legend-items">
        <span class="legend-tag"><i class="legend-color" style="background:var(--ok)"></i> Physical VRAM (NVIDIA-SMI)</span>
        <span class="legend-tag"><i class="legend-color" style="background:#f59e0b"></i> Desktop reserve floor</span>
        <span class="legend-tag"><i class="legend-color" style="background:var(--warn);width:3px;height:10px"></i> Engine / Admin Event</span>
      </div>
    </div>
    <div class="panel-body" style="padding-bottom:12px;">
      <div class="chart-wrapper">
        <canvas id="timeline-canvas" class="chart-canvas" tabindex="0" role="img" aria-label="10 Hz timeline showing Physical VRAM usage and DXGI system budget over time. Use arrow keys to scrub history."></canvas>
        <div id="chart-tooltip" class="chart-tooltip"></div>
      </div>
    </div>
  </section>

  <!-- Main 2-Column Grid -->
  <div class="layout-grid">

    <!-- Column Left: Detailed Telemetry & Speculative Breakdown -->
    <div style="display:flex; flex-direction:column; gap:20px;">
      
      <!-- GPU & KV Sizing Details -->
      <section class="panel" aria-label="Hardware Architecture">
        <div class="panel-header">
          <h2 class="panel-title"><i></i> Hardware &amp; KV Cache Architecture</h2>
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
              <span class="dense-key">Desktop Reserve <span class="help-tip" title="Device memory the engine must leave free. Enforced when the plan is sized AND as a runtime floor: an allocation that would breach it is refused rather than granted.">?</span></span>
              <span id="dt-reserve" class="dense-val">—</span>
            </div>
            <div class="dense-row">
              <span class="dense-key">Engine Reservation <span class="help-tip" title="What the engine planned to hold: weights plus KV pool plus workspace plus the CUDA graph allowance. The allowance sits inside the reservation but is never allocated, so resident bytes are legitimately lower.">?</span></span>
              <span id="dt-reservation" class="dense-val">—</span>
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
            <div style="color:var(--text-muted); font-size:10px; margin-bottom:4px; text-transform:uppercase;">
              Boot Capacity Resolution Line <span class="help-tip" title="Exact parameter ledger resolved during engine initialization (tokens, pages, slack, graph allowance, context cache).">?</span>
            </div>
            <div id="dt-capacity-text">Waiting for engine capacity resolution...</div>
          </div>
        </div>
      </section>

      <!-- Speculative Acceptance Breakdown -->
      <section class="panel" aria-label="MTP Speculative Breakdown">
        <div class="panel-header">
          <h2 class="panel-title"><i></i> Speculative Decoding</h2>
          <span id="dt-mtp-rate-badge" class="kpi-badge badge-ok">—</span>
        </div>
        <div class="panel-body">
          <div class="dense-table">
            <div class="dense-row">
              <span class="dense-key">Speculative Backend</span>
              <span id="dt-mtp-backend" class="dense-val">—</span>
            </div>
            <div class="dense-row">
              <span class="dense-key">Draft Window</span>
              <span id="dt-mtp-window" class="dense-val">—</span>
            </div>
            <div class="dense-row">
              <span class="dense-key">Prefix Cache Reuse Mix <span class="help-tip" title="Breakdown of multi-turn prefix reuse: Seed restore, append, or full reset.">?</span></span>
              <span id="dt-reuse-mix" class="dense-val">—</span>
            </div>
          </div>

          <div style="margin-top:14px;">
            <div style="display:flex; justify-content:space-between; font-size:10px; font-family:var(--font-mono); color:var(--text-muted); text-transform:uppercase; margin-bottom:4px;">
              <span>Acceptance Per Draft Position <span class="help-tip" title="How often the token drafted at each position was accepted. Positions beyond the configured draft window are not shown.">?</span></span>
              <span id="mtp-pos-legend">Position Ratios</span>
            </div>
            <div id="mtp-bars-container" class="mtp-bars">
              <div class="mtp-col"><div class="mtp-bar-track"><div class="mtp-bar"></div></div><span class="mtp-label">P1</span></div>
              <div class="mtp-col"><div class="mtp-bar-track"><div class="mtp-bar"></div></div><span class="mtp-label">P2</span></div>
              <div class="mtp-col"><div class="mtp-bar-track"><div class="mtp-bar"></div></div><span class="mtp-label">P3</span></div>
              <div class="mtp-col"><div class="mtp-bar-track"><div class="mtp-bar"></div></div><span class="mtp-label">P4</span></div>
              <div class="mtp-col"><div class="mtp-bar-track"><div class="mtp-bar"></div></div><span class="mtp-label">P5</span></div>
            </div>
          </div>
        </div>
      </section>

    </div>

    <!-- Column Right: Real-time Terminal Log & Insights -->
    <div style="display:flex; flex-direction:column; gap:20px;">

      <!-- Streaming Engine Console -->
      <section class="panel" aria-label="Engine Console Log">
        <div class="panel-header">
          <h2 class="panel-title"><i></i> Live Engine Log Stream</h2>
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
            <input type="text" id="log-search" class="log-search" placeholder="Filter log..." aria-label="Filter log text" />
          </div>
          <div id="log-content" class="log-scroll-pane" aria-live="polite">Connecting to engine log tail...</div>
        </div>
      </section>

      <!-- Autonomous Diagnostics & Insights -->
      <section class="panel" aria-label="Diagnostic Insights">
        <div class="panel-header">
          <h2 class="panel-title"><i></i> Performance &amp; Cache Insights <span class="help-tip" title="Autonomous log analyzer diagnosing prefix degradation, latency saturation, and cache pressure.">?</span></h2>
          <span id="insights-count-badge" class="kpi-badge badge-info">0 INSIGHTS</span>
        </div>
        <div class="panel-body">
          <div id="insights-container" class="insights-grid">
            <div style="color:var(--text-muted); font-size:12px; font-family:var(--font-mono); text-align:center; padding:16px;">
              Analyzing request logs for prefix collapse, latency saturation, and cache pressure...
            </div>
          </div>
        </div>
      </section>

    </div>

  </div>

  <!-- Engine Configuration -->
  <section class="panel cfg-panel" aria-label="Engine Configuration" style="margin-top:16px;">
    <div class="panel-header">
      <h2 class="panel-title"><i></i> Configuration <span class="help-tip" title="Edits are written to the supervisor's config file and take effect when the engine next starts. The form is generated from the engine's own parameter table, so every field carries the engine's bounds.">?</span></h2>
      <span id="cfg-path" class="kpi-badge badge-info">—</span>
    </div>
    <div class="panel-body">
      <div class="cfg-tabs" role="tablist">
        <button class="log-filter-btn active" data-cfg-tab="network" role="tab">Networking</button>
        <button class="log-filter-btn" data-cfg-tab="api" role="tab">API</button>
        <button class="log-filter-btn" data-cfg-tab="capacity" role="tab">Capacity</button>
        <button class="log-filter-btn" data-cfg-tab="memory" role="tab">Memory</button>
        <button class="log-filter-btn" data-cfg-tab="features" role="tab">Features</button>
        <button class="log-filter-btn" data-cfg-tab="raw" role="tab">Command line</button>
      </div>
      <div id="cfg-body" class="cfg-groups">
        <div style="color:var(--text-muted); font-size:12px; font-family:var(--font-mono); padding:16px;">Loading configuration...</div>
      </div>
      <div id="cfg-errors" class="cfg-errors" style="display:none;"></div>
      <div class="cfg-actions">
        <div id="cfg-status" class="cfg-status">No changes</div>
        <div class="control-group">
          <button id="cfg-revert" class="btn" disabled>Revert</button>
          <button id="cfg-save" class="btn btn-start" disabled>Save</button>
          <button id="cfg-save-restart" class="btn btn-restart" disabled>Save &amp; restart engine</button>
        </div>
      </div>
    </div>
  </section>

</main>

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
)HTML"
R"HTML(
  // State Management
  let lastState = null;
  let autoScroll = true;
  let activeLogFilter = 'all';
  let logSearchQuery = '';
  let rawLogTail = '';
  let keyboardScrubIdx = null;
  let isDocumentVisible = !document.hidden;

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
  const a11yLive = document.getElementById('a11y-live-region');

  // Confirmation Modal Refs
  const confirmModal = document.getElementById('confirm-modal');
  const modalTitle = document.getElementById('modal-title');
  const modalDesc = document.getElementById('modal-desc');
  const modalCancelBtn = document.getElementById('modal-cancel-btn');
  const modalConfirmBtn = document.getElementById('modal-confirm-btn');

  // Canvas High-DPI Setup
  const ctx = canvas.getContext('2d');
  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    if (lastState && lastState.series && isDocumentVisible) {
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
    keyboardScrubIdx = null;
    if (lastState && lastState.series && isDocumentVisible) drawTimeline(lastState.series);
  });
  canvas.addEventListener('mouseleave', () => {
    mousePos = null;
    if (keyboardScrubIdx == null) tooltip.style.display = 'none';
    if (lastState && lastState.series && isDocumentVisible) drawTimeline(lastState.series);
  });

  // Canvas Keyboard Navigation & Screen Reader Announcement
  canvas.addEventListener('keydown', e => {
    if (!lastState || !lastState.series) return;
    const ser = lastState.series;
    const n = (ser.t_ms || []).length;
    if (n < 2) return;

    if (keyboardScrubIdx == null) keyboardScrubIdx = n - 1;

    if (e.key === 'ArrowLeft') {
      keyboardScrubIdx = Math.max(0, keyboardScrubIdx - 1);
      e.preventDefault();
    } else if (e.key === 'ArrowRight') {
      keyboardScrubIdx = Math.min(n - 1, keyboardScrubIdx + 1);
      e.preventDefault();
    } else if (e.key === 'Home') {
      keyboardScrubIdx = 0;
      e.preventDefault();
    } else if (e.key === 'End') {
      keyboardScrubIdx = n - 1;
      e.preventDefault();
    } else {
      return;
    }

    mousePos = null;
    drawTimeline(ser);
  });

  function drawTimeline(ser) {
    if (!ser || !isDocumentVisible) return;
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
    ctx.fillStyle = '#94a3b8';
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

    // The desktop reserve as a threshold line: memory used above it is memory the
    // desktop was promised. A curve of the DXGI budget used to be drawn here; it is
    // blind to other processes, stays collapsed after pressure clears and turns
    // optimistic again near saturation, so it could not be read as a signal.
    const reserveFloor = (lastState && lastState.admin_vram && lastState.admin_vram.desktop_reserve)
      ? lastState.admin_vram.desktop_reserve.runtime_floor_bytes : 0;
    const totalForLine = (lastState && lastState.nvidia_smi) ? lastState.nvidia_smi.total_bytes : 0;
    if (reserveFloor > 0 && totalForLine > 0) {
      const limitY = getY(totalForLine - reserveFloor);
      ctx.beginPath();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#f59e0b';
      ctx.setLineDash([6, 4]);
      ctx.moveTo(getX(0), limitY);
      ctx.lineTo(getX(n - 1), limitY);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#f59e0b';
      ctx.font = '10px ui-monospace, monospace';
      ctx.fillText('desktop reserve floor', getX(0) + 6, limitY - 4);
    }

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

    // Crosshair Scrub (Mouse or Keyboard)
    let activeIdx = null;
    if (mousePos && mousePos.x >= pX && mousePos.x <= W - 16) {
      const ratio = (mousePos.x - pX) / graphW;
      activeIdx = Math.min(n - 1, Math.max(0, Math.round(ratio * (n - 1))));
    } else if (keyboardScrubIdx != null) {
      activeIdx = keyboardScrubIdx;
    }

    if (activeIdx != null) {
      const curX = getX(activeIdx);
      const tDelta = ((t1 - t[activeIdx])/1000).toFixed(1);

      ctx.beginPath();
      ctx.strokeStyle = '#94a3b8';
      ctx.lineWidth = 1;
      ctx.moveTo(curX, pY);
      ctx.lineTo(curX, pY + graphH);
      ctx.stroke();

      // Highlight Points
      ctx.fillStyle = '#10b981';
      ctx.beginPath();
      ctx.arc(curX, getY(u[activeIdx]), 4, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#38bdf8';
      ctx.beginPath();
      ctx.arc(curX, getY(b[activeIdx]), 4, 0, Math.PI * 2);
      ctx.fill();

      // Tooltip HUD
      tooltip.style.display = 'block';
      tooltip.innerHTML = `
        <div style="color:#f8fafc; font-weight:700; margin-bottom:2px;">T - ${tDelta}s</div>
        <div style="color:#10b981">VRAM: ${gib(u[activeIdx])}</div>
        <div style="color:#38bdf8">DXGI: ${gib(b[activeIdx])}</div>
      `;

      if (a11yLive) {
        a11yLive.textContent = `Timeline at T minus ${tDelta}s: Physical VRAM ${gib(u[activeIdx])}, DXGI budget ${gib(b[activeIdx])}`;
      }
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
    if (isDocumentVisible) {
      liveDot.className = 'pulse-dot';
      liveLabel.textContent = s.monitor_only ? 'MONITOR ONLY (UNMANAGED)' : 'STREAMING LIVE (SSE)';
    }

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
    const frac = totalBytes ? (usedBytes / totalBytes) : 0;
    const pct = (frac * 100).toFixed(1);

    document.getElementById('kpi-vram-main').textContent = gib(usedBytes);
    document.getElementById('kpi-vram-pct').textContent = pct + '%';
    document.getElementById('kpi-vram-bar').style.transform = `scaleX(${frac})`;
    document.getElementById('kpi-vram-free').textContent = gib(totalBytes - usedBytes);
    document.getElementById('kpi-vram-total').textContent = gib(totalBytes);

    // The engine's own memory report. Absent means the engine is down or too old
    // to serve /admin/vram; the panel says which rather than showing a stale plan.
    const av = s.admin_vram || null;
    const reserveEl = document.getElementById('dt-reserve');
    const reservationEl = document.getElementById('dt-reservation');
    if (av && av.desktop_reserve) {
      const r = av.desktop_reserve;
      const holding = r.holding !== false;
      reserveEl.textContent = r.runtime_floor_bytes
        ? `${gib(r.runtime_floor_bytes)} floor · ${gib(r.free_bytes)} free`
        : `not enforced · ${gib(r.free_bytes)} free`;
      reserveEl.style.color = holding ? 'var(--ok)' : 'var(--bad)';
      const p = av.plan || {};
      reservationEl.textContent =
        `${gib(p.runtime_reservation_bytes)} planned · ${mib(p.cuda_graph_allowance_bytes)} graph allowance`;
    } else {
      reserveEl.textContent = s.admin_vram_note || '—';
      reserveEl.style.color = 'var(--text-muted)';
      reservationEl.textContent = '—';
    }

    // Hardware Details
    document.getElementById('adapter-tag').textContent = dxgi.adapter_name ? 'RTX PRO 6000' : 'GPU 0';
    document.getElementById('dt-gpu-name').textContent = dxgi.adapter_name || 'NVIDIA GPU';
    document.getElementById('dt-phys-vram').textContent = `${gib(usedBytes)} / ${gib(totalBytes)} (${pct}%)`;
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

    // Speculative telemetry. A configured backend with no drafts yet is a
    // different state from no backend at all, and neither is an acceptance of 0.
    const specBackend = reqs.mtp_backend || '';
    const specOff = !specBackend || specBackend === 'none';
    document.getElementById('dt-mtp-backend').textContent =
      specOff ? 'none — speculation disabled' : specBackend;
    document.getElementById('dt-mtp-window').textContent =
      reqs.mtp_draft_window ? `${reqs.mtp_draft_window} draft token${reqs.mtp_draft_window === 1 ? '' : 's'}` : '—';
    document.getElementById('kpi-mtp-badge').textContent = specOff ? 'OFF' : specBackend.toUpperCase();
    if (specOff) {
      document.getElementById('kpi-mtp-pct').textContent = '—';
      document.getElementById('kpi-mtp-ratio').textContent = 'not configured';
      document.getElementById('kpi-mtp-fallbacks').textContent = '—';
      document.getElementById('dt-mtp-rate-badge').textContent = 'OFF';
    } else if (!reqs.mtp_drafted) {
      document.getElementById('kpi-mtp-pct').textContent = '—';
      document.getElementById('kpi-mtp-ratio').textContent = 'no drafts yet';
      document.getElementById('dt-mtp-rate-badge').textContent = 'IDLE';
    }
    if (reqs.mtp_drafted) {
      const ratePct = ((reqs.mtp_last_accept_rate || 0) * 100).toFixed(1);
      document.getElementById('kpi-mtp-pct').textContent = ratePct + '%';
      document.getElementById('kpi-mtp-ratio').textContent = `${(reqs.mtp_accepted||0).toLocaleString()} / ${(reqs.mtp_drafted||0).toLocaleString()}`;
      document.getElementById('kpi-mtp-fallbacks').textContent = reqs.mtp_fallback_steps || 0;
      document.getElementById('dt-mtp-rate-badge').textContent = `${ratePct}% ACCEPTED`;

      // One bar per configured draft position; extra columns are hidden rather
      // than drawn empty, which would read as "accepted nothing at position 5".
      const pos = reqs.mtp_accepted_per_position || [];
      const maxPos = Math.max(1, ...pos);
      const cols = document.querySelectorAll('#mtp-bars-container .mtp-bar');
      document.querySelectorAll('#mtp-bars-container .mtp-col').forEach((col, idx) => {
        col.style.display = idx < Math.max(pos.length, reqs.mtp_draft_window || 0) ? '' : 'none';
      });
      pos.forEach((pVal, idx) => {
        if (cols[idx]) {
          const ratio = Math.max(0.04, pVal / maxPos);
          cols[idx].style.transform = `scaleY(${ratio})`;
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
    if (autoScroll && isDocumentVisible) {
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
        return `
          <article class="insight-card ${sev}">
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
          </article>
        `;
      }).join('');
    }

    // Timeline Rendering
    if (isDocumentVisible) {
      drawTimeline(s.series);
    }
  }

  // Non-Blocking In-Palette Confirmation Modal Flow with Focus Trap
  let pendingAction = null;
  let lastFocusedTrigger = null;

  function showConfirmModal(actionName, btn) {
    lastFocusedTrigger = btn;
    pendingAction = { actionName, btn };
    modalTitle.textContent = `Confirm Engine ${actionName.toUpperCase()}`;
    modalDesc.textContent = `Are you sure you want to send ${actionName.toUpperCase()} to the resident NInfer inference engine?`;
    confirmModal.style.display = 'flex';
    modalConfirmBtn.focus();
  }

  function hideConfirmModal() {
    confirmModal.style.display = 'none';
    pendingAction = null;
    if (lastFocusedTrigger) {
      lastFocusedTrigger.focus();
      lastFocusedTrigger = null;
    }
  }

  modalCancelBtn.onclick = hideConfirmModal;
  modalConfirmBtn.onclick = async () => {
    if (!pendingAction) return;
    const { actionName, btn } = pendingAction;
    hideConfirmModal();

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
  };

  // Keyboard trap for modal
  confirmModal.addEventListener('keydown', e => {
    if (confirmModal.style.display === 'flex') {
      if (e.key === 'Escape') {
        hideConfirmModal();
        e.preventDefault();
      } else if (e.key === 'Tab') {
        if (e.shiftKey) {
          if (document.activeElement === modalCancelBtn) {
            modalConfirmBtn.focus();
            e.preventDefault();
          }
        } else {
          if (document.activeElement === modalConfirmBtn) {
            modalCancelBtn.focus();
            e.preventDefault();
          }
        }
      }
    }
  });

  btnStart.onclick = () => showConfirmModal('start', btnStart);
  btnStop.onclick = () => showConfirmModal('stop', btnStop);
  btnRestart.onclick = () => showConfirmModal('restart', btnRestart);

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
    }).catch(() => {
      btnCopyLog.textContent = '✗ Error';
      setTimeout(() => { btnCopyLog.textContent = '📋 Copy'; }, 1500);
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

  // Page Visibility Lifecycle: Zero resource waste when tab is in background
  document.addEventListener('visibilitychange', () => {
    isDocumentVisible = !document.hidden;
    if (isDocumentVisible) {
      liveLabel.textContent = lastState && lastState.monitor_only ? 'MONITOR ONLY (UNMANAGED)' : 'STREAMING LIVE (SSE)';
      liveDot.className = 'pulse-dot';
      fetch('/api/state').then(r => r.json()).then(render).catch(() => {});
    } else {
      liveLabel.textContent = 'IDLE (TAB IN BACKGROUND)';
      liveDot.className = 'pulse-dot paused';
    }
  });

  // ---------------------------------------------------------------------
  // Configuration editor.
  //
  // Every field is generated from /api/config's schema, which the C++ parameter
  // table produces. Nothing about a parameter -- its label, bounds, choices or
  // help text -- is written twice, so a knob added in the engine appears here
  // with the engine's own description rather than a copy that drifts.
  //
  // Two things are shown and deliberately not editable: the executable and the
  // working directory. The control gate on this page is a loopback check plus a
  // header, which stops a random web page from driving it -- it is not an
  // authorization system, and it should not be what stands between a browser and
  // choosing which binary runs.
  // ---------------------------------------------------------------------
  const cfgBody = document.getElementById('cfg-body');
  const cfgErrors = document.getElementById('cfg-errors');
  const cfgStatus = document.getElementById('cfg-status');
  const cfgSave = document.getElementById('cfg-save');
  const cfgSaveRestart = document.getElementById('cfg-save-restart');
  const cfgRevert = document.getElementById('cfg-revert');
  const cfgPath = document.getElementById('cfg-path');
  let cfgData = null;
  let cfgTab = 'network';
  const cfgEdits = { params: {}, engine: {}, supervisor: {} };

  function cfgDirtyCount() {
    return Object.keys(cfgEdits.params).length + Object.keys(cfgEdits.engine).length +
           Object.keys(cfgEdits.supervisor).length;
  }

  function cfgSetDirtyUi() {
    const n = cfgDirtyCount();
    cfgStatus.className = 'cfg-status' + (n ? ' dirty' : '');
    cfgStatus.textContent = n ? n + ' unsaved change' + (n === 1 ? '' : 's') : 'No changes';
    const writable = cfgData && cfgData.writable;
    cfgSave.disabled = !n || !writable;
    cfgSaveRestart.disabled = !n || !writable;
    cfgRevert.disabled = !n;
  }

  function cfgEdit(section, key, value, original) {
    // An edit returned to its original value stops being an edit, so the save
    // button reflects what would actually change rather than what was touched.
    const same = typeof original === 'boolean'
      ? value === original
      : String(value) === String(original == null ? '' : original);
    if (same) { delete cfgEdits[section][key]; } else { cfgEdits[section][key] = value; }
    cfgSetDirtyUi();
  }

  function fieldRow(spec, value) {
    const wrap = document.createElement('div');
    wrap.className = 'cfg-field';
    const top = document.createElement('div');
    top.className = 'cfg-field-top';
    const label = document.createElement('span');
    label.className = 'cfg-label';
    label.textContent = spec.label;
    const flag = document.createElement('span');
    flag.className = 'cfg-flag';
    flag.textContent = spec.flag;
    top.appendChild(label);
    top.appendChild(flag);
    wrap.appendChild(top);

    let input;
    if (spec.kind === 'flag') {
      input = document.createElement('input');
      input.type = 'checkbox';
      input.checked = value === 'true';
      const line = document.createElement('label');
      line.className = 'cfg-check';
      line.appendChild(input);
      const t = document.createElement('span');
      t.textContent = input.checked ? 'enabled' : 'disabled';
      line.appendChild(t);
      input.onchange = () => {
        t.textContent = input.checked ? 'enabled' : 'disabled';
        cfgEdit('params', spec.key, input.checked, value === 'true');
        input.classList.toggle('dirty', input.checked !== (value === 'true'));
      };
      wrap.appendChild(line);
    } else if (spec.kind === 'enum') {
      input = document.createElement('select');
      input.className = 'cfg-select';
      // A parameter with choices is still optional: "not set" is a real state and
      // is not the same as the engine's default, which the engine picks itself.
      const none = document.createElement('option');
      none.value = '';
      none.textContent = '(not set)';
      input.appendChild(none);
      spec.choices.forEach(c => {
        const o = document.createElement('option');
        o.value = c;
        o.textContent = c;
        input.appendChild(o);
      });
      input.value = value || '';
      input.onchange = () => {
        cfgEdit('params', spec.key, input.value, value || '');
        input.classList.toggle('dirty', input.value !== (value || ''));
      };
      wrap.appendChild(input);
    } else {
      input = document.createElement('input');
      input.className = 'cfg-input';
      input.type = spec.kind === 'int' ? 'number' : 'text';
      if (spec.kind === 'int') { input.min = spec.min; input.max = spec.max; }
      if (spec.kind === 'int_or_auto') { input.placeholder = spec.min + '..' + spec.max + ' or auto'; }
      input.value = value || '';
      input.oninput = () => {
        cfgEdit('params', spec.key, input.value, value || '');
        input.classList.toggle('dirty', input.value !== (value || ''));
      };
      wrap.appendChild(input);
    }

    const help = document.createElement('div');
    help.className = 'cfg-help';
    help.textContent = (spec.kind === 'int' || spec.kind === 'int_or_auto')
      ? spec.help + ' Range ' + spec.min + '-' + spec.max + '.'
      : spec.help;
    wrap.appendChild(help);
    return wrap;
  }

  function plainField(section, key, label, value, opts) {
    opts = opts || {};
    const wrap = document.createElement('div');
    wrap.className = 'cfg-field';
    const top = document.createElement('div');
    top.className = 'cfg-field-top';
    const l = document.createElement('span');
    l.className = 'cfg-label';
    l.textContent = label;
    top.appendChild(l);
    if (opts.hint) {
      const h = document.createElement('span');
      h.className = 'cfg-flag';
      h.textContent = opts.hint;
      top.appendChild(h);
    }
    wrap.appendChild(top);
    let input;
    if (opts.type === 'bool') {
      input = document.createElement('input');
      input.type = 'checkbox';
      input.checked = !!value;
      const line = document.createElement('label');
      line.className = 'cfg-check';
      line.appendChild(input);
      const t = document.createElement('span');
      t.textContent = input.checked ? 'enabled' : 'disabled';
      line.appendChild(t);
      input.onchange = () => {
        t.textContent = input.checked ? 'enabled' : 'disabled';
        cfgEdit(section, key, input.checked, !!value);
        input.classList.toggle('dirty', input.checked !== !!value);
      };
      wrap.appendChild(line);
    } else {
      input = document.createElement('input');
      input.className = 'cfg-input';
      input.type = opts.type === 'int' ? 'number' : 'text';
      input.value = value == null ? '' : value;
      if (opts.readonly) {
        input.readOnly = true;
      } else {
        input.oninput = () => {
          const v = opts.type === 'int' ? parseInt(input.value, 10) : input.value;
          cfgEdit(section, key, v, value);
          input.classList.toggle('dirty', String(input.value) !== String(value == null ? '' : value));
        };
      }
      wrap.appendChild(input);
    }
    if (opts.help) {
      const help = document.createElement('div');
      help.className = 'cfg-help';
      help.textContent = opts.help;
      wrap.appendChild(help);
    }
    return wrap;
  }

  function cfgNote(text) {
    const n = document.createElement('div');
    n.className = 'cfg-note';
    n.textContent = text;
    return n;
  }

  function renderConfig() {
    if (!cfgData) { return; }
    cfgBody.innerHTML = '';
    const eng = cfgData.engine || {};
    const sup = cfgData.supervisor || {};

    if (cfgTab === 'network') {
      cfgBody.appendChild(plainField('engine', 'engine_host', 'Engine host', eng.engine_host,
        { help: 'Address the engine listens on. The supervisor polls health here.' }));
      cfgBody.appendChild(plainField('engine', 'engine_port', 'Engine port', eng.engine_port,
        { type: 'int', help: 'Port the engine serves its OpenAI-compatible API on.' }));
      cfgBody.appendChild(plainField('engine', 'device', 'GPU device index', eng.device,
        { type: 'int', help: 'Which CUDA device the engine runs on.' }));
      cfgBody.appendChild(plainField('supervisor', 'port', 'Dashboard port', sup.port,
        { type: 'int', help: 'This page. Changing it needs the supervisor restarted, not just the engine.' }));
      cfgBody.appendChild(plainField('supervisor', 'host', 'Dashboard host', sup.host,
        { help: 'Must be a loopback address unless binding beyond loopback is enabled.' }));
      cfgBody.appendChild(plainField('supervisor', 'bind_any', 'Bind beyond loopback', sup.bind_any,
        { type: 'bool', help: 'Exposes engine start, stop and this configuration form to your network.' }));
      cfgBody.appendChild(cfgNote('The dashboard reaches the engine over loopback. Binding beyond loopback exposes engine lifecycle control and this form to anything that can reach the port; the only checks are a loopback peer test and a request header, which stop a stray web page rather than a determined caller.'));
    } else if (cfgTab === 'api') {
      cfgBody.appendChild(plainField('engine', 'api_key_file', 'API key file', eng.api_key_file,
        { help: eng.api_key_present
            ? 'A key was read from this file. The key itself is never sent to this page.'
            : 'No key could be read from this path, so the engine accepts unauthenticated requests.' }));
      cfgBody.appendChild(plainField('engine', 'request_log', 'Request log path', eng.request_log,
        { help: 'Per-request JSONL. This is what fills the throughput, reuse and speculation panels above.' }));
      (cfgData.schema || []).filter(x => x.group === 'api').forEach(spec => {
        cfgBody.appendChild(fieldRow(spec, (cfgData.params || {})[spec.key]));
      });
      cfgBody.appendChild(cfgNote('The API key is read from the file above by the supervisor and handed to the engine. It never appears on this page, in /api/config, or in the argument list under "Command line".'));
    } else if (cfgTab === 'raw') {
      cfgBody.appendChild(plainField('engine', '_executable', 'Executable', eng.executable,
        { readonly: true, hint: 'file-only', help: 'Not editable here by design: this decides which binary runs.' }));
      cfgBody.appendChild(plainField('engine', '_workdir', 'Working directory', eng.workdir,
        { readonly: true, hint: 'file-only' }));
      cfgBody.appendChild(plainField('engine', '_artifact', 'Model artifact', eng.artifact,
        { readonly: true, hint: 'file-only' }));
      const pre = document.createElement('pre');
      pre.className = 'insight-evidence';
      pre.style.gridColumn = '1/-1';
      pre.textContent = (eng.args || []).join(' ');
      cfgBody.appendChild(pre);
      if ((eng.passthrough || []).length) {
        cfgBody.appendChild(cfgNote('Passed through unchanged because this build does not describe them: ' +
          eng.passthrough.join(' ') + '. They survive an edit made here.'));
      }
    } else {
      const group = cfgTab;
      (cfgData.schema || []).filter(x => x.group === group).forEach(spec => {
        cfgBody.appendChild(fieldRow(spec, (cfgData.params || {})[spec.key]));
      });
      if (group === 'memory') {
        cfgBody.appendChild(cfgNote('Quantization changes what the model outputs, not only what it costs. On this model the FP8 output head diverges from BF16 after about 95 greedy tokens and the FP8 KV cache diverges later. Measure before trusting a saving.'));
      }
    }
    cfgSetDirtyUi();
  }

  async function cfgPost(payload) {
    const res = await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-NInfer-Supervisor': '1' },
      body: JSON.stringify(payload)
    });
    let body = {};
    try { body = await res.json(); } catch (e) {}
    return { ok: res.ok, status: res.status, body: body };
  }

  function showCfgErrors(list) {
    if (!list || !list.length) { cfgErrors.style.display = 'none'; return; }
    cfgErrors.style.display = 'block';
    cfgErrors.innerHTML = '<strong>The configuration was not saved.</strong><ul>' +
      list.map(e => '<li>' + esc(e) + '</li>').join('') + '</ul>';
  }

  async function cfgApply(thenRestart) {
    showCfgErrors(null);
    const payload = {
      params: cfgEdits.params,
      engine: cfgEdits.engine,
      supervisor: cfgEdits.supervisor
    };
    const r = await cfgPost(payload);
    if (!r.ok) {
      showCfgErrors(r.body.details || [r.body.error || ('HTTP ' + r.status)]);
      return;
    }
    cfgData = r.body;
    cfgEdits.params = {};
    cfgEdits.engine = {};
    cfgEdits.supervisor = {};
    renderConfig();
    cfgStatus.className = 'cfg-status saved';
    const needsEngine = r.body.engine_restart_required;
    const needsSup = r.body.supervisor_restart_required;
    cfgStatus.textContent = 'Saved' +
      (needsEngine ? ' - takes effect when the engine restarts' : '') +
      (needsSup ? ' - dashboard host or port needs the supervisor restarted' : '');
    if (thenRestart && needsEngine) {
      await fetch('/api/restart', { method: 'POST', headers: { 'X-NInfer-Supervisor': '1' } });
      cfgStatus.textContent = 'Saved. Engine restarting - the model reloads, which takes about a minute.';
    }
  }

  document.querySelectorAll('[data-cfg-tab]').forEach(b => {
    b.onclick = () => {
      document.querySelectorAll('[data-cfg-tab]').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      cfgTab = b.dataset.cfgTab;
      renderConfig();
    };
  });
  cfgSave.onclick = () => cfgApply(false);
  cfgSaveRestart.onclick = () => cfgApply(true);
  cfgRevert.onclick = () => {
    cfgEdits.params = {};
    cfgEdits.engine = {};
    cfgEdits.supervisor = {};
    showCfgErrors(null);
    renderConfig();
  };

  fetch('/api/config').then(r => r.json()).then(d => {
    cfgData = d;
    cfgPath.textContent = d.config_path ? d.config_path.split(/[\\/]/).pop() : 'no config file';
    cfgPath.title = d.config_path || '';
    const portEl = document.getElementById('kpi-engine-port');
    if (portEl) {
      portEl.textContent = (d.engine && d.engine.engine_port) ? ':' + d.engine.engine_port : '-';
    }
    const sub = document.getElementById('brand-sub');
    if (sub) {
      sub.textContent = 'PRECISION MISSION CONTROL // PORT ' + (location.port || '80');
    }
    renderConfig();
  }).catch(() => {
    cfgBody.innerHTML = '<div style="color:var(--text-muted); font-size:12px; padding:16px;">Configuration is unavailable: this page is not being served over loopback.</div>';
  });

  // SSE & Initial Fetch
  const es = new EventSource('/api/events');
  es.onmessage = e => {
    try {
      if (isDocumentVisible) {
        render(JSON.parse(e.data));
      }
    } catch (err) {}
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
