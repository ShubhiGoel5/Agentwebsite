"""FastAPI Web UI Server and Trace Viewer for Autonomous Task Worker."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Ensure root workspace is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from worker.adapters.openrouter import OpenRouterLLMAdapter
from worker.adapters.scripted import ScriptedLLMAdapter
from worker.core.llm import FinalAnswer, ToolCall
from worker.core.loop import AgentLoop, TaskConfig
from worker.core.policy import PolicyGate
from worker.core.trace import TraceLogger
from worker.env.mock_erp import MockERPEnvironment
from worker.env.mock_mail import MockMailEnvironment
from worker.tasks.customer_update import create_customer_update_task
from worker.tasks.expense_recon import create_expense_recon_task
from worker.tasks.invoice import create_invoice_task

app = FastAPI(title="Autonomous Task Worker Dashboard & Trace Viewer")

# Global state for current execution trace & pending approval
active_runs: Dict[str, Dict[str, Any]] = {}


class RunTaskRequest(BaseModel):
    task_type: str  # "invoice", "expense", "customer"
    adapter_mode: str  # "scripted", "openrouter"
    api_key: Optional[str] = ""
    auto_approve: bool = True


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autonomous Task Worker | Control Center & Trace Viewer</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0b0f19;
            --bg-secondary: #111827;
            --bg-card: rgba(17, 24, 39, 0.7);
            --bg-card-hover: rgba(31, 41, 55, 0.8);
            --border-color: rgba(255, 255, 255, 0.08);
            --border-accent: rgba(59, 130, 246, 0.3);
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
            --accent-amber: #f59e0b;
            --accent-red: #ef4444;
            --accent-purple: #8b5cf6;
            --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg-primary);
            color: var(--text-main);
            font-family: var(--font-sans);
            min-height: 100vh;
            line-height: 1.5;
            overflow-x: hidden;
        }

        /* Glassmorphism background glow */
        .glow-bg {
            position: fixed;
            top: -150px;
            right: -150px;
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, rgba(59, 130, 246, 0.15) 0%, rgba(139, 92, 246, 0.05) 50%, transparent 70%);
            z-index: -1;
            pointer-events: none;
        }
        .glow-bg-left {
            position: fixed;
            bottom: -150px;
            left: -150px;
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, rgba(16, 185, 129, 0.12) 0%, transparent 70%);
            z-index: -1;
            pointer-events: none;
        }

        header {
            border-bottom: 1px solid var(--border-color);
            background: rgba(11, 15, 25, 0.8);
            backdrop-filter: blur(12px);
            position: sticky;
            top: 0;
            z-index: 100;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo-group {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .logo-icon {
            width: 38px;
            height: 38px;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            color: #fff;
            box-shadow: 0 4px 14px rgba(59, 130, 246, 0.4);
        }

        .logo-title {
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            background: linear-gradient(to right, #ffffff, #93c5fd);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .badge-pill {
            background: rgba(59, 130, 246, 0.15);
            border: 1px solid rgba(59, 130, 246, 0.3);
            color: var(--accent-blue);
            font-size: 0.75rem;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            font-weight: 600;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
            display: grid;
            grid-template-columns: 360px 1fr;
            gap: 2rem;
        }

        /* Controls Panel */
        .panel {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(16px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }

        .panel-title {
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 1.25rem;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .form-group {
            margin-bottom: 1.25rem;
        }

        label {
            display: block;
            font-size: 0.825rem;
            font-weight: 500;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        select, input[type="text"], input[type="password"] {
            width: 100%;
            background: rgba(17, 24, 39, 0.9);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 0.75rem 1rem;
            border-radius: 10px;
            font-family: inherit;
            font-size: 0.9rem;
            transition: all 0.2s ease;
        }

        select:focus, input:focus {
            outline: none;
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
        }

        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            cursor: pointer;
        }

        .checkbox-group input {
            width: auto;
        }

        .btn {
            width: 100%;
            background: linear-gradient(135deg, var(--accent-blue), #2563eb);
            color: white;
            font-weight: 600;
            padding: 0.85rem 1.25rem;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 0.95rem;
            transition: all 0.2s ease;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        /* Metrics bar */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            padding: 1.25rem;
            border-radius: 14px;
            text-align: center;
        }

        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            margin-top: 0.25rem;
            font-family: var(--font-mono);
        }

        .status-badge {
            display: inline-block;
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .status-idle { background: rgba(156, 163, 175, 0.15); color: #9ca3af; }
        .status-running { background: rgba(59, 130, 246, 0.15); color: var(--accent-blue); }
        .status-success { background: rgba(16, 185, 129, 0.15); color: var(--accent-green); }
        .status-escalated { background: rgba(239, 68, 68, 0.15); color: var(--accent-red); }

        /* Result Card */
        .result-card {
            background: rgba(16, 185, 129, 0.07);
            border: 1px solid rgba(16, 185, 129, 0.25);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            display: none;
        }
        .result-card.error {
            background: rgba(239, 68, 68, 0.07);
            border-color: rgba(239, 68, 68, 0.25);
        }
        .result-card.paused {
            background: rgba(245, 158, 11, 0.07);
            border-color: rgba(245, 158, 11, 0.25);
        }
        .result-icon { font-size: 2rem; margin-bottom: 0.5rem; }
        .result-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #fff;
            margin-bottom: 0.5rem;
        }
        .result-body {
            font-size: 0.95rem;
            color: #d1fae5;
            line-height: 1.6;
        }
        .result-card.error .result-body { color: #fca5a5; }
        .result-card.paused .result-body { color: #fde68a; }
        .result-data-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-top: 1rem;
        }
        .result-data-pill {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 0.35rem 0.75rem;
            font-size: 0.82rem;
            color: var(--text-muted);
        }
        .result-data-pill strong { color: var(--text-main); }

        /* Step Timeline */
        .steps-container {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            min-height: 500px;
        }
        .steps-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
        }
        .timeline {
            display: flex;
            flex-direction: column;
            gap: 0;
        }
        .step-row {
            display: flex;
            gap: 1rem;
            padding: 0.85rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.04);
            align-items: flex-start;
            animation: fadeSlideIn 0.3s ease forwards;
        }
        .step-row:last-child { border-bottom: none; }
        @keyframes fadeSlideIn {
            from { opacity: 0; transform: translateY(6px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .step-icon {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            flex-shrink: 0;
            margin-top: 0.1rem;
        }
        .step-icon.start    { background: rgba(59,130,246,0.15); }
        .step-icon.tool     { background: rgba(139,92,246,0.15); }
        .step-icon.success  { background: rgba(16,185,129,0.15); }
        .step-icon.warning  { background: rgba(245,158,11,0.15); }
        .step-icon.error    { background: rgba(239,68,68,0.15); }
        .step-icon.verify   { background: rgba(56,189,248,0.15); }
        .step-icon.ask      { background: rgba(245,158,11,0.15); }
        .step-body { flex: 1; min-width: 0; }
        .step-label {
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-muted);
            margin-bottom: 0.2rem;
        }
        .step-desc {
            font-size: 0.95rem;
            font-weight: 500;
            color: var(--text-main);
            line-height: 1.4;
        }
        .step-time {
            font-size: 0.75rem;
            color: var(--text-muted);
            white-space: nowrap;
            margin-top: 0.2rem;
        }
        .step-detail-btn {
            background: transparent;
            border: 1px solid var(--border-color);
            color: var(--text-muted);
            padding: 0.15rem 0.5rem;
            border-radius: 5px;
            font-size: 0.72rem;
            cursor: pointer;
            margin-top: 0.4rem;
            transition: all 0.15s;
        }
        .step-detail-btn:hover { border-color: var(--accent-blue); color: var(--accent-blue); }
        .step-detail-raw {
            display: none;
            margin-top: 0.6rem;
            background: rgba(0,0,0,0.4);
            border-radius: 8px;
            padding: 0.75rem;
            font-family: var(--font-mono);
            font-size: 0.78rem;
            color: #a7f3d0;
            overflow-x: auto;
            white-space: pre-wrap;
        }
        .empty-state {
            text-align: center;
            padding: 4rem 2rem;
            color: var(--text-muted);
        }
    </style>
</head>
<body>
    <div class="glow-bg"></div>
    <div class="glow-bg-left"></div>

    <header>
        <div class="logo-group">
            <div class="logo-icon">⚡</div>
            <div>
                <div class="logo-title">Autonomous Task Worker</div>
                <div style="font-size: 0.75rem; color: var(--text-muted);">OpenRouter Harness & Reliability Engine</div>
            </div>
        </div>
        <div class="badge-pill">Python 3.11+ Core Engine</div>
    </header>

    <div class="container">
        <!-- Controls Left Sidebar -->
        <div class="panel">
            <div class="panel-title">⚙️ Task Config & Adapter</div>
            
            <div class="form-group">
                <label for="taskSelect">Target Task Blueprint</label>
                <select id="taskSelect">
                    <option value="invoice">Invoice to ERP Processing (Email parsing -> ERP write -> Read-back)</option>
                    <option value="expense">Expense Reconciliation (CSV vs ERP ledger anomaly scan)</option>
                    <option value="customer">Customer Record Update (Search -> Edit email -> Read-back)</option>
                </select>
            </div>

            <div class="form-group">
                <label for="adapterSelect">LLM Adapter Mode</label>
                <select id="adapterSelect" onchange="toggleApiKeyInput()">
                    <option value="scripted">Scripted Fake Model (Deterministic Test Script - Free)</option>
                    <option value="openrouter">OpenRouter API Adapter (Real OpenRouter Free Models)</option>
                </select>
            </div>

            <div class="form-group" id="apiKeyGroup" style="display: none;">
                <label for="apiKey">OpenRouter API Key</label>
                <input type="password" id="apiKey" placeholder="sk-or-v1-...">
            </div>

            <button class="btn" id="runBtn" onclick="startTaskExecution()">
                <span>🚀 Launch Task Loop</span>
            </button>

            <div style="margin-top: 2rem; border-top: 1px solid var(--border-color); padding-top: 1.25rem;">
                <div class="panel-title" style="font-size: 0.9rem;">🛡️ Reliability Guarantees</div>
                <ul style="font-size: 0.8rem; color: var(--text-muted); padding-left: 1.2rem; display: flex; flex-direction: column; gap: 0.5rem;">
                    <li>Domain-agnostic agent loop</li>
                    <li>3-strikes schema validation recovery</li>
                    <li>Independent verifier read-back checks</li>
                    <li>Automatic 503 transient retries</li>
                    <li>Risk-based policy approval gate</li>
                </ul>
            </div>
        </div>

        <!-- Main Workspace Right -->
        <div>
            <!-- Metrics header -->
            <div class="metrics-grid">
                <div class="metric-card">
                    <div style="font-size: 0.75rem; color: var(--text-muted);">EXECUTION STATUS</div>
                    <div class="metric-value">
                        <span id="statusBadge" class="status-badge status-idle">IDLE</span>
                    </div>
                </div>
                <div class="metric-card">
                    <div style="font-size: 0.75rem; color: var(--text-muted);">STEPS TAKEN</div>
                    <div class="metric-value" id="stepsValue">0</div>
                </div>
                <div class="metric-card">
                    <div style="font-size: 0.75rem; color: var(--text-muted);">RELIABILITY STRIKES</div>
                    <div class="metric-value" id="strikesValue" style="color: var(--accent-green);">0 / 3</div>
                </div>
                <div class="metric-card">
                    <div style="font-size: 0.75rem; color: var(--text-muted);">VERIFIER STATUS</div>
                    <div class="metric-value" id="verifierValue" style="font-size: 1rem; margin-top: 0.5rem; color: var(--text-muted);">UNCHECKED</div>
                </div>
            </div>

            <!-- Result Summary Card -->
            <div class="result-card" id="resultCard">
                <div class="result-icon" id="resultIcon">✅</div>
                <div class="result-title" id="resultTitle">Task Complete</div>
                <div class="result-body" id="resultBody"></div>
                <div class="result-data-grid" id="resultDataGrid"></div>
            </div>

            <!-- Step-by-step Timeline -->
            <div class="steps-container">
                <div class="steps-header">
                    <div style="font-weight: 600; font-size: 1.1rem; color: #fff;">🪄 What the Agent Did</div>
                    <button style="background: transparent; border: 1px solid var(--border-color); color: var(--text-muted); padding: 0.35rem 0.75rem; border-radius: 6px; cursor: pointer; font-size: 0.8rem;" onclick="clearTrace()">Clear</button>
                </div>
                <div class="timeline" id="timeline">
                    <div class="empty-state">
                        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">⚙️</div>
                        <div>No task run yet.</div>
                        <div style="font-size: 0.85rem; margin-top: 0.25rem;">Select a task on the left and click "Launch Task Loop".</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function toggleApiKeyInput() {
            const mode = document.getElementById('adapterSelect').value;
            document.getElementById('apiKeyGroup').style.display = (mode === 'openrouter') ? 'block' : 'none';
        }

        // ── Human-readable labels for each event type ──────────────────────
        const EVENT_META = {
            task_start:          { icon: '🚀', cls: 'start',   label: 'Task Started' },
            llm_request:         null,   // hide — internal plumbing
            llm_error:           { icon: '❌', cls: 'error',   label: 'Model Error' },
            tool_call_decided:   { icon: '🔧', cls: 'tool',    label: 'Action Chosen' },
            policy_check:        { icon: '🛡️', cls: 'warning', label: 'Safety Check' },
            transient_retry:     { icon: '🔄', cls: 'warning', label: 'Retrying...' },
            tool_execution:      { icon: '⚡', cls: 'tool',    label: 'Action Executed' },
            final_claim:         { icon: '📋', cls: 'verify',  label: 'Result Submitted' },
            verification_check:  { icon: '🔍', cls: 'verify',  label: 'Verifying Result' },
            strike_count:        { icon: '⚠️', cls: 'error',   label: 'Validation Issue' },
            loop_detected:       { icon: '🔁', cls: 'error',   label: 'Loop Detected' },
            ask_user:            { icon: '❓', cls: 'ask',     label: 'Needs Your Input' },
            task_end:            { icon: '✅', cls: 'success', label: 'Task Complete' },
            escalated:           { icon: '🚨', cls: 'error',   label: 'Escalated' },
            max_steps_exceeded:  { icon: '⏱️', cls: 'error',   label: 'Step Limit Reached' },
        };

        function clearTrace() {
            document.getElementById('timeline').innerHTML = `
                <div class="empty-state">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">⚙️</div>
                    <div>No task run yet.</div>
                </div>`;
            document.getElementById('resultCard').style.display = 'none';
            document.getElementById('statusBadge').className = 'status-badge status-idle';
            document.getElementById('statusBadge').innerText = 'IDLE';
            document.getElementById('stepsValue').innerText = '0';
            document.getElementById('strikesValue').innerText = '0 / 3';
            document.getElementById('verifierValue').innerText = 'UNCHECKED';
            document.getElementById('verifierValue').style.color = 'var(--text-muted)';
        }

        async function startTaskExecution() {
            const taskType = document.getElementById('taskSelect').value;
            const adapterMode = document.getElementById('adapterSelect').value;
            const apiKey = document.getElementById('apiKey').value;

            const runBtn = document.getElementById('runBtn');
            runBtn.disabled = true;
            runBtn.innerHTML = '<span>⏳ Executing Task Loop...</span>';

            document.getElementById('statusBadge').className = 'status-badge status-running';
            document.getElementById('statusBadge').innerText = 'RUNNING';

            try {
                const response = await fetch('/api/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        task_type: taskType,
                        adapter_mode: adapterMode,
                        api_key: apiKey
                    })
                });

                const result = await response.json();
                renderExecutionResult(result);
            } catch (err) {
                alert('Execution Request Failed: ' + err.message);
                document.getElementById('statusBadge').className = 'status-badge status-escalated';
                document.getElementById('statusBadge').innerText = 'ERROR';
            } finally {
                runBtn.disabled = false;
                runBtn.innerHTML = '<span>🚀 Launch Task Loop</span>';
            }
        }

        function renderExecutionResult(res) {
            // ── Metric badges ─────────────────────────────────────────────
            document.getElementById('stepsValue').innerText = res.steps_taken;

            const badge = document.getElementById('statusBadge');
            badge.innerText = res.status.toUpperCase();
            if (res.status === 'success') badge.className = 'status-badge status-success';
            else if (res.status === 'escalated') badge.className = 'status-badge status-escalated';
            else badge.className = 'status-badge status-running';

            const verifierElem = document.getElementById('verifierValue');
            if (res.status === 'success') {
                verifierElem.innerText = 'VERIFIED ✓';
                verifierElem.style.color = 'var(--accent-green)';
            } else {
                verifierElem.innerText = res.status.toUpperCase();
                verifierElem.style.color = 'var(--accent-red)';
            }

            // ── Result summary card ───────────────────────────────────────
            const rc = document.getElementById('resultCard');
            rc.style.display = 'block';
            rc.className = 'result-card';

            if (res.status === 'success' && res.final_answer) {
                document.getElementById('resultIcon').innerText  = '✅';
                document.getElementById('resultTitle').innerText = 'Task Completed Successfully';
                document.getElementById('resultBody').innerText  = res.final_answer.content;
                const grid = document.getElementById('resultDataGrid');
                grid.innerHTML = '';
                const data = res.final_answer.result_data || {};
                Object.entries(data).forEach(([k, v]) => {
                    const pill = document.createElement('div');
                    pill.className = 'result-data-pill';
                    pill.innerHTML = `<strong>${k.replace(/_/g,' ')}:</strong> ${JSON.stringify(v)}`;
                    grid.appendChild(pill);
                });
            } else if (res.status === 'paused_for_user') {
                rc.classList.add('paused');
                document.getElementById('resultIcon').innerText  = '❓';
                document.getElementById('resultTitle').innerText = 'Agent Needs Your Input';
                document.getElementById('resultBody').innerText  = res.reason;
                document.getElementById('resultDataGrid').innerHTML = '';
            } else {
                rc.classList.add('error');
                document.getElementById('resultIcon').innerText  = '❌';
                document.getElementById('resultTitle').innerText = 'Task Could Not Complete';
                document.getElementById('resultBody').innerText  = res.reason;
                document.getElementById('resultDataGrid').innerHTML = '';
            }

            // ── Human-readable step timeline ──────────────────────────────
            const timeline = document.getElementById('timeline');
            timeline.innerHTML = '';
            let detailCounter = 0;

            if (res.events && res.events.length > 0) {
                res.events.forEach(evt => {
                    const meta = EVENT_META[evt.event_type];
                    if (!meta) return;   // skip hidden events (e.g. llm_request)

                    const timeStr = new Date(evt.timestamp * 1000).toLocaleTimeString();
                    const dataStr = Object.keys(evt.data).length > 0
                        ? JSON.stringify(evt.data, null, 2) : '';
                    const detailId = 'detail-' + (detailCounter++);

                    // Build a plain-English description
                    let desc = evt.message;
                    if (evt.event_type === 'tool_call_decided' && evt.data.name) {
                        desc = `Calling tool: "${evt.data.name}"`;
                        if (evt.data.args && Object.keys(evt.data.args).length) {
                            const argsStr = Object.entries(evt.data.args)
                                .map(([k,v]) => `${k}: ${JSON.stringify(v)}`).join(', ');
                            desc += ` (${argsStr})`;
                        }
                    } else if (evt.event_type === 'tool_execution') {
                        desc = evt.data.success
                            ? `✓ Tool ran successfully`
                            : `✗ Tool failed: ${evt.data.error || 'unknown error'}`;
                    } else if (evt.event_type === 'verification_check') {
                        desc = evt.data.passed
                            ? `Result verified — ${evt.data.reason}`
                            : `Verification failed — ${evt.data.reason}`;
                    } else if (evt.event_type === 'policy_check') {
                        desc = `Safety check for "${evt.data.decision || ''}" — ${evt.data.reason || evt.message}`;
                    } else if (evt.event_type === 'final_claim') {
                        desc = evt.data.content || evt.message;
                    }

                    const row = document.createElement('div');
                    row.className = 'step-row';
                    row.innerHTML = `
                        <div class="step-icon ${meta.cls}">${meta.icon}</div>
                        <div class="step-body">
                            <div class="step-label">${meta.label}</div>
                            <div class="step-desc">${desc}</div>
                            <div class="step-time">${timeStr} &nbsp;·&nbsp; Step ${evt.step}</div>
                            ${dataStr ? `<button class="step-detail-btn" onclick="toggleDetail('${detailId}')">Show details</button>
                            <pre class="step-detail-raw" id="${detailId}">${dataStr}</pre>` : ''}
                        </div>`;
                    timeline.appendChild(row);
                });
            }

            if (timeline.children.length === 0) {
                timeline.innerHTML = '<div class="empty-state"><div>No steps recorded.</div></div>';
            }
        }

        function toggleDetail(id) {
            const el = document.getElementById(id);
            const btn = el.previousElementSibling;
            if (el.style.display === 'block') {
                el.style.display = 'none';
                btn.innerText = 'Show details';
            } else {
                el.style.display = 'block';
                btn.innerText = 'Hide details';
            }
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_CONTENT


@app.post("/api/run")
def run_task_api(req: RunTaskRequest):
    mail_env = MockMailEnvironment()
    erp_env = MockERPEnvironment()

    if req.task_type == "invoice":
        task_config = create_invoice_task(mail_env=mail_env, erp_env=erp_env)
        default_script = [
            ToolCall(name="read_email", args={"email_id": "msg_001"}),
            ToolCall(
                name="parse_pdf_text",
                args={
                    "raw_text": "INVOICE NUMBER: INV-2026-9912\nVENDOR: Acme Supplies Inc.\nVENDOR_ID: VEND-ACME-01\nAMOUNT: 4250.00\nCURRENCY: USD\nITEMS: Office Furniture (10), Printers (2)\nDUE_DATE: 2026-10-31"
                },
            ),
            ToolCall(
                name="post_erp_invoice",
                args={
                    "invoice_number": "INV-2026-9912",
                    "vendor_id": "VEND-ACME-01",
                    "amount": 4250.0,
                    "currency": "USD",
                },
            ),
            FinalAnswer(
                content="Invoice #INV-2026-9912 for $4,250.00 processed from email msg_001 and posted to ERP.",
                result_data={"invoice_number": "INV-2026-9912", "amount": 4250.0},
            ),
        ]
    elif req.task_type == "expense":
        task_config = create_expense_recon_task(erp_env=erp_env)
        default_script = [
            ToolCall(name="read_csv_file", args={"csv_content": task_config.initial_facts["csv_data"]}),
            ToolCall(name="get_erp_expenses", args={}),
            FinalAnswer(
                content="Reconciliation finished. Flagged 2 mismatches.",
                result_data={
                    "mismatches": [
                        {"description": "Team Lunch", "csv_amount": 95.50, "erp_amount": 85.50},
                        {"description": "Taxi fare", "csv_amount": 35.00, "erp_amount": None},
                    ]
                },
            ),
        ]
    else:  # customer
        task_config = create_customer_update_task(erp_env=erp_env)
        default_script = [
            ToolCall(name="get_customer", args={"customer_id": "CUST-101"}),
            ToolCall(
                name="update_customer_record",
                args={
                    "customer_id": "CUST-101",
                    "field_name": "email",
                    "new_value": "apexpay@apextech.com",
                },
            ),
            FinalAnswer(
                content="Customer CUST-101 email updated to apexpay@apextech.com",
                result_data={"customer_id": "CUST-101", "email": "apexpay@apextech.com"},
            ),
        ]

    # Select LLM Client
    if req.adapter_mode == "openrouter":
        api_key = req.api_key or os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise HTTPException(status_code=400, detail="OPENROUTER_API_KEY is required for OpenRouter mode.")
        llm_client = OpenRouterLLMAdapter(api_key=api_key)
    else:
        llm_client = ScriptedLLMAdapter(script=default_script)

    trace_logger = TraceLogger()

    # Web auto-approval policy callback if auto_approve is set
    def web_approval_cb(tool_name, args, risk, reason):
        return req.auto_approve

    policy_gate = PolicyGate(
        approval_rules=task_config.approval_rules,
        approval_callback=web_approval_cb,
    )

    loop = AgentLoop(
        config=task_config,
        llm_client=llm_client,
        policy_gate=policy_gate,
        trace_logger=trace_logger,
    )

    res = loop.run()

    return {
        "status": res.status,
        "steps_taken": res.steps_taken,
        "reason": res.reason,
        "final_answer": res.final_answer.to_dict() if res.final_answer else None,
        "events": trace_logger.get_events(),
    }


def start_server(port: int = 8000):
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    start_server()
