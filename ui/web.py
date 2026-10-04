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

        /* Timeline Event Stream */
        .trace-container {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            min-height: 500px;
        }

        .trace-header {
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
            gap: 1rem;
        }

        .event-card {
            background: rgba(17, 24, 39, 0.5);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            position: relative;
            transition: border-color 0.2s;
        }

        .event-card:hover {
            border-color: var(--border-accent);
        }

        .event-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }

        .event-type {
            font-size: 0.8rem;
            font-weight: 700;
            font-family: var(--font-mono);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .type-task_start { color: var(--accent-blue); }
        .type-tool_call_decided { color: var(--accent-purple); }
        .type-tool_execution { color: var(--accent-green); }
        .type-policy_check { color: var(--accent-amber); }
        .type-verification_check { color: #38bdf8; }
        .type-task_end { color: var(--accent-green); }
        .type-strike_count { color: var(--accent-red); }

        .event-msg {
            font-size: 0.95rem;
            font-weight: 500;
            color: var(--text-main);
            margin-bottom: 0.5rem;
        }

        .event-data {
            background: rgba(0, 0, 0, 0.4);
            border-radius: 8px;
            padding: 0.75rem;
            font-family: var(--font-mono);
            font-size: 0.8rem;
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

            <!-- Trace Event Log Stream -->
            <div class="trace-container">
                <div class="trace-header">
                    <div style="font-weight: 600; font-size: 1.1rem; color: #fff;">📜 Live Execution Event Trace (JSONL Log)</div>
                    <button style="background: transparent; border: 1px solid var(--border-color); color: var(--text-muted); padding: 0.35rem 0.75rem; border-radius: 6px; cursor: pointer; font-size: 0.8rem;" onclick="clearTrace()">Clear Timeline</button>
                </div>

                <div class="timeline" id="timeline">
                    <div class="empty-state">
                        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">⚙️</div>
                        <div>No active task execution running yet.</div>
                        <div style="font-size: 0.85rem; margin-top: 0.25rem;">Select a task blueprint on the left and click "Launch Task Loop".</div>
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

        function clearTrace() {
            document.getElementById('timeline').innerHTML = `
                <div class="empty-state">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">⚙️</div>
                    <div>No active task execution running yet.</div>
                </div>`;
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
            document.getElementById('stepsValue').innerText = res.steps_taken;
            
            const badge = document.getElementById('statusBadge');
            badge.innerText = res.status.toUpperCase();
            if (res.status === 'success') badge.className = 'status-badge status-success';
            else if (res.status === 'escalated') badge.className = 'status-badge status-escalated';
            else badge.className = 'status-badge status-running';

            const verifierElem = document.getElementById('verifierValue');
            if (res.status === 'success') {
                verifierElem.innerText = 'VERIFIED PASSED';
                verifierElem.style.color = 'var(--accent-green)';
            } else {
                verifierElem.innerText = res.status.toUpperCase();
                verifierElem.style.color = 'var(--accent-red)';
            }

            const timeline = document.getElementById('timeline');
            timeline.innerHTML = '';

            if (res.events && res.events.length > 0) {
                res.events.forEach(evt => {
                    const card = document.createElement('div');
                    card.className = 'event-card';

                    const dateStr = new Date(evt.timestamp * 1000).toLocaleTimeString();
                    const dataStr = Object.keys(evt.data).length > 0 ? JSON.stringify(evt.data, null, 2) : '';

                    card.innerHTML = `
                        <div class="event-header">
                            <span class="event-type type-\${evt.event_type}">[Step \${evt.step}] \${evt.event_type}</span>
                            <span style="font-size: 0.75rem; color: var(--text-muted); font-family: var(--font-mono);">\${dateStr}</span>
                        </div>
                        <div class="event-msg">\${evt.message}</div>
                        \${dataStr ? `<pre class="event-data">\${dataStr}</pre>` : ''}
                    `;
                    timeline.appendChild(card);
                });
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
