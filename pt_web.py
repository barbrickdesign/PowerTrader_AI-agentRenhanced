"""
PowerTrader AI - Mobile Web Dashboard
======================================
A Flask-based web dashboard that provides a mobile-responsive interface
for monitoring and controlling the PowerTrader AI trading bot.

Run alongside pt_hub.py or standalone (headless):
    python pt_web.py [--port 5000] [--host 0.0.0.0]

Then open http://<your-ip>:5000 on any browser/mobile device.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template_string, request, redirect, url_for

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SETTINGS_FILE = os.path.join(BASE_DIR, "gui_settings.json")
HUB_DATA_DIR_DEFAULT = os.path.join(BASE_DIR, "hub_data")

DEFAULT_SETTINGS: Dict[str, Any] = {
    "main_neural_dir": "",
    "coins": ["BTC", "ETH", "XRP", "BNB", "DOGE"],
    "trade_start_level": 3,
    "start_allocation_pct": 0.005,
    "dca_multiplier": 2.0,
    "dca_levels": [-2.5, -5.0, -10.0, -20.0, -30.0, -40.0, -50.0],
    "max_dca_buys_per_24h": 2,
    "pm_start_pct_no_dca": 5.0,
    "pm_start_pct_with_dca": 2.5,
    "trailing_gap_pct": 0.5,
    "hub_data_dir": "",
    "script_neural_runner2": "pt_thinker.py",
    "script_neural_trainer": "pt_trainer.py",
    "script_trader": "pt_trader.py",
    "auto_start_scripts": False,
}

# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

_procs: Dict[str, Optional[subprocess.Popen]] = {
    "neural": None,
    "trader": None,
}
_proc_lock = threading.Lock()


def _load_settings() -> Dict[str, Any]:
    try:
        if os.path.isfile(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
            merged = dict(DEFAULT_SETTINGS)
            merged.update(data)
            merged["coins"] = [
                str(c).strip().upper()
                for c in merged.get("coins", [])
                if str(c).strip()
            ]
            hub_dir = str(merged.get("hub_data_dir") or "").strip()
            if not hub_dir:
                hub_dir = HUB_DATA_DIR_DEFAULT
            merged["hub_data_dir"] = hub_dir
            return merged
    except Exception:
        pass
    s = dict(DEFAULT_SETTINGS)
    s["hub_data_dir"] = HUB_DATA_DIR_DEFAULT
    return s


def _hub_data_dir() -> str:
    return str(_load_settings().get("hub_data_dir") or HUB_DATA_DIR_DEFAULT)


def _safe_read_json(path: str) -> Optional[dict]:
    try:
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    try:
        if not os.path.isfile(path):
            return rows
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return rows


def _read_int_file(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return int(float(fh.read().strip()))
    except Exception:
        return 0


def _build_coin_folders(main_dir: str, coins: List[str]) -> Dict[str, str]:
    folders: Dict[str, str] = {}
    if not main_dir or not os.path.isdir(main_dir):
        return folders
    for coin in coins:
        coin = coin.upper().strip()
        if coin == "BTC":
            folders[coin] = main_dir
        else:
            folders[coin] = os.path.join(main_dir, coin)
    return folders


def _fmt_money(x: Any) -> str:
    try:
        v = float(x)
        if abs(v) >= 1000:
            return f"${v:,.2f}"
        return f"${v:.2f}"
    except Exception:
        return "N/A"


def _fmt_price(x: Any) -> str:
    try:
        p = float(x)
    except Exception:
        return "N/A"
    if p == 0:
        return "0"
    ap = abs(p)
    if ap >= 1.0:
        decimals = 2
    else:
        decimals = int(-math.floor(math.log10(ap))) + 3
        decimals = max(2, min(12, decimals))
    s = f"{p:.{decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _fmt_pct(x: Any) -> str:
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return "N/A"


def _proc_is_running(name: str) -> bool:
    with _proc_lock:
        p = _procs.get(name)
        return bool(p and p.poll() is None)


def _start_proc(name: str, script: str) -> str:
    with _proc_lock:
        p = _procs.get(name)
        if p and p.poll() is None:
            return f"{name} already running"
        path = os.path.join(BASE_DIR, script)
        if not os.path.isfile(path):
            return f"Script not found: {path}"
        try:
            env = dict(os.environ)
            proc = subprocess.Popen(
                [sys.executable, path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=BASE_DIR,
                env=env,
            )
            _procs[name] = proc
            return f"{name} started (pid {proc.pid})"
        except Exception as exc:
            return f"Failed to start {name}: {exc}"


def _stop_proc(name: str) -> str:
    with _proc_lock:
        p = _procs.get(name)
        if not p or p.poll() is not None:
            return f"{name} not running"
        try:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
            _procs[name] = None
            return f"{name} stopped"
        except Exception as exc:
            return f"Failed to stop {name}: {exc}"


# ---------------------------------------------------------------------------
# API data builders
# ---------------------------------------------------------------------------

def _get_status_data() -> Dict[str, Any]:
    settings = _load_settings()
    hub_dir = settings["hub_data_dir"]
    main_dir = str(settings.get("main_neural_dir") or "").strip()
    if not main_dir or not os.path.isabs(main_dir):
        main_dir = os.path.join(BASE_DIR, main_dir) if main_dir else BASE_DIR
    coins = settings.get("coins") or []

    # ---- account / positions ----
    trader_status = _safe_read_json(os.path.join(hub_dir, "trader_status.json")) or {}
    acct = trader_status.get("account") or {}
    positions_raw = trader_status.get("positions") or {}

    # ---- P&L ledger ----
    pnl_data = _safe_read_json(os.path.join(hub_dir, "pnl_ledger.json")) or {}
    realized_pnl = float(pnl_data.get("realized_pnl_usd", 0) or 0)
    trade_count = int(pnl_data.get("trade_count", 0) or 0)

    # ---- neural signals ----
    coin_folders = _build_coin_folders(main_dir, coins)
    neural_signals: List[Dict[str, Any]] = []
    for coin in coins:
        folder = coin_folders.get(coin, "")
        long_sig = 0
        short_sig = 0
        if folder and os.path.isdir(folder):
            long_path = os.path.join(folder, "long_dca_signal.txt")
            if os.path.isfile(long_path):
                long_sig = _read_int_file(long_path)
            short_path = os.path.join(folder, "short_dca_signal.txt")
            if os.path.isfile(short_path):
                short_sig = _read_int_file(short_path)
            else:
                mem_path = os.path.join(folder, "memory.json")
                if os.path.isfile(mem_path):
                    obj = _safe_read_json(mem_path) or {}
                    try:
                        short_sig = int(float(obj.get("short_dca_signal", 0)))
                    except Exception:
                        short_sig = 0
        in_trade = False
        pos = positions_raw.get(coin) or {}
        try:
            in_trade = float(pos.get("quantity", 0) or 0) > 0
        except Exception:
            in_trade = False
        neural_signals.append({
            "coin": coin,
            "long": long_sig,
            "short": short_sig,
            "in_trade": in_trade,
        })

    # ---- active positions (only non-zero qty) ----
    positions: List[Dict[str, Any]] = []
    for sym, pos in positions_raw.items():
        try:
            qty = float(pos.get("quantity", 0) or 0)
        except Exception:
            qty = 0
        if qty <= 0:
            continue
        positions.append({
            "coin": sym,
            "qty": _fmt_price(qty),
            "value": _fmt_money(pos.get("value_usd")),
            "avg_cost": _fmt_price(pos.get("avg_cost_basis")),
            "buy_price": _fmt_price(pos.get("current_buy_price")),
            "buy_pnl": _fmt_pct(pos.get("gain_loss_pct_buy")),
            "sell_price": _fmt_price(pos.get("current_sell_price")),
            "sell_pnl": _fmt_pct(pos.get("gain_loss_pct_sell")),
            "dca_stages": int(pos.get("dca_triggered_stages", 0) or 0),
            "next_dca": str(pos.get("next_dca_display") or ""),
            "trail_line": _fmt_price(pos.get("trail_line")),
        })

    # ---- recent trade history (last 50) ----
    history_path = os.path.join(hub_dir, "trade_history.jsonl")
    history_rows = _read_jsonl(history_path)
    history_rows = history_rows[-50:]
    history: List[Dict[str, Any]] = []
    for row in reversed(history_rows):
        try:
            ts = float(row.get("ts", 0) or 0)
            ts_str = time.strftime("%m/%d %H:%M", time.localtime(ts)) if ts else ""
        except Exception:
            ts_str = ""
        history.append({
            "ts": ts_str,
            "symbol": str(row.get("symbol") or ""),
            "side": str(row.get("side") or ""),
            "tag": str(row.get("tag") or ""),
            "qty": _fmt_price(row.get("qty")),
            "price": _fmt_price(row.get("price")),
            "total": _fmt_money(row.get("total_usd")),
            "pnl": _fmt_pct(row.get("pnl_pct")) if row.get("side", "").lower() == "sell" else "",
        })

    # ---- process states ----
    neural_running = _proc_is_running("neural")
    trader_running = _proc_is_running("trader")

    # ---- account value sparkline (last 60 points) ----
    av_path = os.path.join(hub_dir, "account_value_history.jsonl")
    av_rows = _read_jsonl(av_path)
    sparkline: List[float] = []
    for r in av_rows[-60:]:
        try:
            sparkline.append(float(r.get("total_account_value", 0) or 0))
        except Exception:
            pass

    return {
        "ts": time.time(),
        "ts_str": time.strftime("%H:%M:%S"),
        "account": {
            "total_value": _fmt_money(acct.get("total_account_value")),
            "holdings_value": _fmt_money(acct.get("holdings_sell_value")),
            "buying_power": _fmt_money(acct.get("buying_power")),
            "pct_in_trade": _fmt_pct(acct.get("percent_in_trade")),
        },
        "pnl": {
            "realized": _fmt_money(realized_pnl),
            "trade_count": trade_count,
        },
        "neural_signals": neural_signals,
        "positions": positions,
        "history": history,
        "processes": {
            "neural_running": neural_running,
            "trader_running": trader_running,
        },
        "sparkline": sparkline,
        "settings": {
            "coins": coins,
            "trade_start_level": settings.get("trade_start_level", 3),
        },
    }


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

# ---------------------------------------------------------------------------
# HTML template (single-file, mobile-first, Bootstrap 5 dark)
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#070B10">
<title>PowerTrader AI</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
      crossorigin="anonymous">
<style>
  :root {
    --pt-bg:       #070B10;
    --pt-bg2:      #0B1220;
    --pt-panel:    #0E1626;
    --pt-panel2:   #121C2F;
    --pt-border:   #243044;
    --pt-fg:       #C7D1DB;
    --pt-muted:    #8B949E;
    --pt-green:    #00FF66;
    --pt-cyan:     #00E5FF;
    --pt-orange:   #FF9900;
    --pt-red:      #FF4444;
  }
  html, body { background: var(--pt-bg); color: var(--pt-fg); font-family: 'Segoe UI', system-ui, sans-serif; }
  .navbar { background: var(--pt-bg2) !important; border-bottom: 1px solid var(--pt-border); }
  .navbar-brand { color: var(--pt-green) !important; font-weight: 700; letter-spacing: .05em; }
  .status-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .dot-green { background: var(--pt-green); }
  .dot-red   { background: var(--pt-red); }
  .card { background: var(--pt-panel); border: 1px solid var(--pt-border); border-radius: 10px; }
  .card-header { background: var(--pt-panel2); border-bottom: 1px solid var(--pt-border);
                 color: var(--pt-cyan); font-size: .8rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }
  .kv-label { color: var(--pt-muted); font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; }
  .kv-value { color: var(--pt-fg); font-size: 1.05rem; font-weight: 500; }
  .kv-value.accent { color: var(--pt-green); }
  .badge-long  { background: #1a3a6a !important; color: var(--pt-cyan) !important; }
  .badge-short { background: #3a2a00 !important; color: var(--pt-orange) !important; }
  .badge-trade { background: var(--pt-green); color: #000 !important; }
  .table { --bs-table-bg: transparent; --bs-table-striped-bg: rgba(255,255,255,.02);
           --bs-table-hover-bg: rgba(0,229,255,.06); font-size: .82rem; }
  .table th { color: var(--pt-muted); font-weight: 600; border-color: var(--pt-border); font-size: .72rem; text-transform: uppercase; }
  .table td { border-color: var(--pt-border); vertical-align: middle; }
  .buy-side  { color: var(--pt-green); font-weight: 600; }
  .sell-side { color: var(--pt-orange); font-weight: 600; }
  .pos-pnl-pos { color: var(--pt-green); }
  .pos-pnl-neg { color: var(--pt-red); }
  .signal-bar { display: inline-flex; gap: 3px; vertical-align: middle; }
  .signal-bar .seg { width: 10px; height: 24px; border-radius: 3px; background: var(--pt-panel2); border: 1px solid var(--pt-border); }
  .signal-bar .seg.lit-long  { background: #3060d0; }
  .signal-bar .seg.lit-short { background: #c06000; }
  svg.sparkline { width: 100%; height: 40px; }
  .ctrl-btn { min-width: 100px; }
  .refresh-spinner { width: 14px; height: 14px; }
  #toast-wrap { position: fixed; bottom: 1rem; right: 1rem; z-index: 9999; }
  footer { border-top: 1px solid var(--pt-border); color: var(--pt-muted); font-size: .72rem; }
  @media (max-width: 576px) {
    .kv-value { font-size: .95rem; }
    .ctrl-btn { min-width: 80px; font-size: .82rem; }
    .table { font-size: .75rem; }
  }
</style>
</head>
<body>

<!-- Navbar -->
<nav class="navbar navbar-dark sticky-top px-3 py-2">
  <span class="navbar-brand">⚡ PowerTrader AI</span>
  <div class="d-flex align-items-center gap-2">
    <span id="proc-neural" class="badge rounded-pill bg-secondary">Neural</span>
    <span id="proc-trader" class="badge rounded-pill bg-secondary">Trader</span>
    <div id="refresh-indicator" title="Refreshing…" class="d-none">
      <svg class="refresh-spinner" viewBox="0 0 24 24" fill="none" stroke="var(--pt-cyan)" stroke-width="2.5"
           stroke-linecap="round" stroke-linejoin="round">
        <polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-.49-3"/>
      </svg>
    </div>
    <span id="last-update" class="text-muted" style="font-size:.7rem">—</span>
  </div>
</nav>

<div class="container-fluid py-3 px-2 px-md-4" style="max-width:1200px;margin:0 auto">

  <!-- Row 1: Account Summary + Controls -->
  <div class="row g-2 mb-2">
    <!-- Account summary -->
    <div class="col-12 col-md-7">
      <div class="card h-100">
        <div class="card-header">Account Summary</div>
        <div class="card-body p-2">
          <div class="row g-2">
            <div class="col-6 col-sm-3">
              <div class="kv-label">Total Value</div>
              <div class="kv-value accent" id="acct-total">—</div>
            </div>
            <div class="col-6 col-sm-3">
              <div class="kv-label">Holdings</div>
              <div class="kv-value" id="acct-holdings">—</div>
            </div>
            <div class="col-6 col-sm-3">
              <div class="kv-label">Buying Power</div>
              <div class="kv-value" id="acct-bp">—</div>
            </div>
            <div class="col-6 col-sm-3">
              <div class="kv-label">% In Trade</div>
              <div class="kv-value" id="acct-pct">—</div>
            </div>
            <div class="col-6 col-sm-3">
              <div class="kv-label">Realized P&L</div>
              <div class="kv-value" id="pnl-realized">—</div>
            </div>
            <div class="col-6 col-sm-3">
              <div class="kv-label">Closed Trades</div>
              <div class="kv-value" id="pnl-count">—</div>
            </div>
          </div>
          <!-- Sparkline -->
          <svg class="sparkline mt-2 d-none" id="sparkline-svg" viewBox="0 0 300 40" preserveAspectRatio="none"></svg>
        </div>
      </div>
    </div>
    <!-- Controls -->
    <div class="col-12 col-md-5">
      <div class="card h-100">
        <div class="card-header">Bot Controls</div>
        <div class="card-body p-2 d-flex flex-column gap-2">
          <div class="d-flex flex-wrap gap-2">
            <button class="btn btn-sm btn-outline-success ctrl-btn" onclick="ctrlAction('start_all')">▶ Start All</button>
            <button class="btn btn-sm btn-outline-danger ctrl-btn"  onclick="ctrlAction('stop_all')">■ Stop All</button>
          </div>
          <div class="d-flex flex-wrap gap-2">
            <button class="btn btn-sm btn-outline-info ctrl-btn"    onclick="ctrlAction('start_neural')">▶ Neural</button>
            <button class="btn btn-sm btn-outline-warning ctrl-btn" onclick="ctrlAction('stop_neural')">■ Neural</button>
          </div>
          <div class="d-flex flex-wrap gap-2">
            <button class="btn btn-sm btn-outline-info ctrl-btn"    onclick="ctrlAction('start_trader')">▶ Trader</button>
            <button class="btn btn-sm btn-outline-warning ctrl-btn" onclick="ctrlAction('stop_trader')">■ Trader</button>
          </div>
          <div id="ctrl-msg" class="text-muted" style="font-size:.75rem;min-height:1.2rem"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Row 2: Neural Signals -->
  <div class="card mb-2">
    <div class="card-header d-flex justify-content-between align-items-center">
      <span>Neural Signals</span>
      <span id="neural-last" class="text-muted" style="font-size:.7rem">Last: —</span>
    </div>
    <div class="card-body p-2">
      <div id="neural-grid" class="d-flex flex-wrap gap-2"></div>
    </div>
  </div>

  <!-- Row 3: Active Positions -->
  <div class="card mb-2">
    <div class="card-header">Active Positions</div>
    <div class="card-body p-0">
      <div class="table-responsive">
        <table class="table table-sm table-hover mb-0">
          <thead><tr>
            <th>Coin</th><th>Qty</th><th>Value</th><th>Avg Cost</th>
            <th>Buy Px</th><th>Buy P&L</th><th>DCA</th><th>Next DCA</th><th>Trail</th>
          </tr></thead>
          <tbody id="pos-tbody"><tr><td colspan="9" class="text-center text-muted py-3">No active positions</td></tr></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Row 4: Trade History -->
  <div class="card mb-2">
    <div class="card-header">Recent Trades (last 50)</div>
    <div class="card-body p-0">
      <div class="table-responsive">
        <table class="table table-sm table-hover mb-0">
          <thead><tr>
            <th>Time</th><th>Symbol</th><th>Side</th><th>Tag</th>
            <th>Qty</th><th>Price</th><th>Total</th><th>P&L</th>
          </tr></thead>
          <tbody id="hist-tbody"><tr><td colspan="8" class="text-center text-muted py-3">No trades yet</td></tr></tbody>
        </table>
      </div>
    </div>
  </div>

</div><!-- /container -->

<footer class="text-center py-2 mt-2">
  PowerTrader AI &bull; Mobile Dashboard &bull; Apache 2.0
</footer>

<!-- Toast -->
<div id="toast-wrap"></div>

<script>
// ---- helpers ----
function showToast(msg, ok) {
  const wrap = document.getElementById('toast-wrap');
  const div = document.createElement('div');
  div.className = 'toast align-items-center text-bg-' + (ok ? 'success' : 'danger') + ' border-0 show mb-1';
  div.style.cssText = 'min-width:220px;max-width:320px;font-size:.82rem';
  div.innerHTML = '<div class="d-flex"><div class="toast-body">' + escHtml(msg) + '</div>'
    + '<button type="button" class="btn-close btn-close-white me-2 m-auto" onclick="this.closest(\'.toast\').remove()"></button></div>';
  wrap.appendChild(div);
  setTimeout(() => div.remove(), 4000);
}
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function pnlClass(s) {
  if (!s || s === 'N/A') return '';
  const n = parseFloat(s);
  return isNaN(n) ? '' : (n >= 0 ? 'pos-pnl-pos' : 'pos-pnl-neg');
}

// ---- signal bars ----
function makeSignalBar(level, cssClass, maxLevels) {
  const segs = maxLevels - 1;  // number of displayable segments (maxLevels includes level-0 as "unlit")
  let html = '<div class="signal-bar">';
  for (let i = 0; i < segs; i++) {
    const lit = (i < level) ? ' ' + cssClass : '';
    html += '<div class="seg' + lit + '"></div>';
  }
  html += '</div>';
  return html;
}

// ---- sparkline ----
function drawSparkline(data) {
  const svg = document.getElementById('sparkline-svg');
  if (!data || data.length < 2) { svg.classList.add('d-none'); return; }
  svg.classList.remove('d-none');
  const W = 300, H = 40;
  const mn = Math.min(...data), mx = Math.max(...data);
  const rng = mx - mn || 1;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * W;
    const y = H - ((v - mn) / rng) * (H - 4) - 2;
    return x + ',' + y;
  }).join(' ');
  svg.innerHTML = '<polyline points="' + pts + '" fill="none" stroke="var(--pt-green)" stroke-width="1.5" stroke-linejoin="round"/>';
}

// ---- neural grid ----
function renderNeuralGrid(signals, startLevel) {
  const grid = document.getElementById('neural-grid');
  grid.innerHTML = '';
  signals.forEach(s => {
    const inTrade = s.in_trade;
    const card = document.createElement('div');
    card.style.cssText = 'background:var(--pt-panel2);border:1px solid var(--pt-border);border-radius:8px;padding:6px 10px;text-align:center;min-width:90px';
    const tradeTag = inTrade ? '<span class="badge badge-trade ms-1" style="font-size:.62rem">IN TRADE</span>' : '';
    const longReady = (s.long >= startLevel && s.short === 0);
    const coinColor = longReady ? 'var(--pt-green)' : 'var(--pt-fg)';
    card.innerHTML =
      '<div style="font-size:.82rem;font-weight:700;color:' + coinColor + '">' + escHtml(s.coin) + tradeTag + '</div>'
      + '<div class="d-flex justify-content-center gap-2 mt-1">'
      + '<div title="LONG">' + makeSignalBar(s.long, 'lit-long', 8) + '<div style="font-size:.62rem;color:var(--pt-cyan)">L:' + s.long + '</div></div>'
      + '<div title="SHORT">' + makeSignalBar(s.short, 'lit-short', 8) + '<div style="font-size:.62rem;color:var(--pt-orange)">S:' + s.short + '</div></div>'
      + '</div>';
    grid.appendChild(card);
  });
}

// ---- positions table ----
function renderPositions(positions) {
  const tbody = document.getElementById('pos-tbody');
  if (!positions || positions.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" class="text-center text-muted py-3">No active positions</td></tr>';
    return;
  }
  tbody.innerHTML = positions.map(p => {
    return '<tr>'
      + '<td><strong>' + escHtml(p.coin) + '</strong></td>'
      + '<td>' + escHtml(p.qty) + '</td>'
      + '<td>' + escHtml(p.value) + '</td>'
      + '<td>' + escHtml(p.avg_cost) + '</td>'
      + '<td>' + escHtml(p.buy_price) + '</td>'
      + '<td class="' + pnlClass(p.buy_pnl) + '">' + escHtml(p.buy_pnl) + '</td>'
      + '<td>' + escHtml(String(p.dca_stages)) + '</td>'
      + '<td style="font-size:.72rem">' + escHtml(p.next_dca) + '</td>'
      + '<td>' + escHtml(p.trail_line) + '</td>'
      + '</tr>';
  }).join('');
}

// ---- history table ----
function renderHistory(rows) {
  const tbody = document.getElementById('hist-tbody');
  if (!rows || rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" class="text-center text-muted py-3">No trades yet</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map(r => {
    const sideClass = r.side.toLowerCase() === 'buy' ? 'buy-side' : 'sell-side';
    return '<tr>'
      + '<td style="white-space:nowrap;font-size:.72rem">' + escHtml(r.ts) + '</td>'
      + '<td>' + escHtml(r.symbol) + '</td>'
      + '<td class="' + sideClass + '">' + escHtml(r.side.toUpperCase()) + '</td>'
      + '<td style="font-size:.7rem">' + escHtml(r.tag) + '</td>'
      + '<td>' + escHtml(r.qty) + '</td>'
      + '<td>' + escHtml(r.price) + '</td>'
      + '<td>' + escHtml(r.total) + '</td>'
      + '<td class="' + pnlClass(r.pnl) + '">' + escHtml(r.pnl) + '</td>'
      + '</tr>';
  }).join('');
}

// ---- process badges ----
function updateProcBadges(procs) {
  const nn = document.getElementById('proc-neural');
  const nt = document.getElementById('proc-trader');
  nn.className = 'badge rounded-pill ' + (procs.neural_running ? 'bg-success' : 'bg-secondary');
  nn.textContent = 'Neural';
  nt.className = 'badge rounded-pill ' + (procs.trader_running ? 'bg-success' : 'bg-secondary');
  nt.textContent = 'Trader';
}

// ---- main refresh ----
let _refreshing = false;
async function refresh() {
  if (_refreshing) return;
  _refreshing = true;
  document.getElementById('refresh-indicator').classList.remove('d-none');
  try {
    const resp = await fetch('/api/status');
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const d = await resp.json();

    // account
    document.getElementById('acct-total').textContent    = d.account.total_value;
    document.getElementById('acct-holdings').textContent = d.account.holdings_value;
    document.getElementById('acct-bp').textContent       = d.account.buying_power;
    document.getElementById('acct-pct').textContent      = d.account.pct_in_trade;
    document.getElementById('pnl-realized').textContent  = d.pnl.realized;
    document.getElementById('pnl-count').textContent     = d.pnl.trade_count;
    document.getElementById('last-update').textContent   = d.ts_str;

    // sparkline
    drawSparkline(d.sparkline);

    // neural
    renderNeuralGrid(d.neural_signals, d.settings.trade_start_level);
    document.getElementById('neural-last').textContent = 'Last: ' + d.ts_str;

    // positions
    renderPositions(d.positions);

    // history
    renderHistory(d.history);

    // process badges
    updateProcBadges(d.processes);

  } catch (e) {
    document.getElementById('last-update').textContent = 'Error';
  } finally {
    _refreshing = false;
    document.getElementById('refresh-indicator').classList.add('d-none');
  }
}

// ---- controls ----
async function ctrlAction(action) {
  const msgEl = document.getElementById('ctrl-msg');
  msgEl.textContent = 'Working…';
  try {
    const resp = await fetch('/api/control', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action}),
    });
    const d = await resp.json();
    const ok = resp.ok && !d.error;
    msgEl.textContent = d.message || d.error || '✓';
    showToast(d.message || d.error || '✓', ok);
    setTimeout(refresh, 1200);
  } catch (e) {
    msgEl.textContent = 'Error: ' + e.message;
    showToast('Error: ' + e.message, false);
  }
}

// ---- auto refresh ----
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(_HTML)


@app.route("/api/status")
def api_status():
    try:
        return jsonify(_get_status_data())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/control", methods=["POST"])
def api_control():
    data = request.get_json(silent=True) or {}
    action = str(data.get("action") or "").strip().lower()
    settings = _load_settings()

    if action == "start_neural":
        msg = _start_proc("neural", settings.get("script_neural_runner2", "pt_thinker.py"))
    elif action == "stop_neural":
        msg = _stop_proc("neural")
    elif action == "start_trader":
        msg = _start_proc("trader", settings.get("script_trader", "pt_trader.py"))
    elif action == "stop_trader":
        msg = _stop_proc("trader")
    elif action == "start_all":
        m1 = _start_proc("neural", settings.get("script_neural_runner2", "pt_thinker.py"))
        m2 = _start_proc("trader", settings.get("script_trader", "pt_trader.py"))
        msg = f"{m1} | {m2}"
    elif action == "stop_all":
        m1 = _stop_proc("neural")
        m2 = _stop_proc("trader")
        msg = f"{m1} | {m2}"
    else:
        return jsonify({"error": f"Unknown action: {action}"}), 400

    return jsonify({"message": msg})


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PowerTrader AI Mobile Web Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()
    print(f"[PowerTrader Web] Starting on http://{args.host}:{args.port}")
    print(f"[PowerTrader Web] Open http://localhost:{args.port} in a browser or on your phone.")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
