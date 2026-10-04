# Autonomous Task Worker (OpenRouter)

An autonomous, reliable AI task execution harness built in Python 3.11+. The engine separates domain knowledge into tools/task configs while handling validation, retries, loop detection, risk policy gates, and independent state verifiers in code.

```
            ┌──────────────────────────────────────────────┐
 User ───▶  │  CLI / Thin Web UI (Trace Viewer, Approvals) │
            └──────────────────┬───────────────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │     Agent Loop      │  observe → decide → gate → act → record
                    └──┬────┬────┬────┬───┘
         ┌─────────────┘    │    │    └──────────────┐
         ▼                  ▼    ▼                   ▼
  ┌────────────┐   ┌────────────┐ ┌─────────────┐ ┌──────────────┐
  │ LLMClient  │   │  Memory    │ │ Policy Gate │ │  Verifier    │
  │ (interface)│   │ (facts,    │ │ (risk tags, │ │ (success     │
  └─────┬──────┘   │  history)  │ │  approval)  │ │  criteria +  │
   ┌────┴─────┐    └────────────┘ └──────┬──────┘ │  read-back)  │
   ▼          ▼                          ▼        └──────┬───────┘
 OpenRouter  Scripted             ┌─────────────┐        │
 adapter     adapter (tests)      │Tool Registry│◀───────┘
                                  └──────┬──────┘
                       ┌─────────────────┼──────────────────┐
                       ▼                 ▼                  ▼
                 mail tools         erp tools          file tools
                 (mock API)        (mock API w/        (CSV/PDF
                                    fault injection)    readers)
                                        │
                                  ┌─────┴──────┐
                                  │ Trace Log  │ (JSONL)
                                  └────────────┘
```

---

## 1. Core Architecture & Design Rationale

1. **Core knows nothing about invoices**: Domain knowledge is completely encapsulated in tool schemas and task configs. `worker/core/` remains strictly domain-agnostic.
2. **Reliability built into the harness**:
   - **3-Strikes Schema Validation**: Malformed model outputs return detailed JSON schema error observations to the LLM. 3 consecutive strikes trigger an escalation.
   - **Transient Fault Recovery**: Harness automatically retries 503s and timeouts with exponential backoff before feeding errors back to the model.
   - **Loop Detection**: Repetitive calls with identical parameters 3 times consecutively trigger loop escalation.
   - **Independent Verifier**: Model completion (`FinalAnswer`) claims are not trusted blindly; an independent read-back query validates real system state.
   - **Risk-Based Policy Gate**: Actions tagged as `irreversible` require explicit user approval.
3. **Free-Tier Budget Protection**:
   - Includes a deterministic `ScriptedLLMAdapter` for zero-cost automated tests.
   - `OpenRouterLLMAdapter` features automatic fallback model rotation (`meta-llama/llama-3.3-70b-instruct:free`, `qwen/qwen-2.5-coder-32b-instruct:free`, `google/gemini-2.0-flash-exp:free`) with 429 rate limit backoff and daily request counters.
4. **Comprehensive Event Tracing**: Every execution step is logged to structured `.jsonl` trace files.

---

## 2. Directory Structure

```
worker/
  core/
    loop.py        # Agent execution loop (domain-agnostic)
    llm.py         # LLMClient protocol & action contracts (ToolCall, FinalAnswer, AskUser)
    memory.py      # Fact store & compact context window builder
    policy.py      # Policy gate & risk level checks (read, write, irreversible)
    verifier.py    # Independent success criteria & read-back verifier
    registry.py    # Tool dataclass, JSON Schema validation, execution safety
    trace.py       # JSONL event logger
  adapters/
    openrouter.py  # OpenRouter API client, model rotation fallback, 429 backoff
    scripted.py    # Scripted fake model adapter for unit & integration tests
  tools/
    mail.py        # Email reading & attachment tools
    erp.py         # ERP invoice, customer, and expense tools
    files.py       # CSV reader & PDF text parsing tools
  env/
    mock_mail.py   # Mock Mail server state machine
    mock_erp.py    # Mock ERP database with fault injection (503s)
  tasks/
    invoice.py         # Task 1: Email Invoice to ERP processing
    expense_recon.py   # Task 2: CSV vs ERP Expense Reconciliation
    customer_update.py # Task 3: Customer Record Update & Read-Back
  ui/
    cli.py         # Interactive Command Line Interface
    web.py         # FastAPI Web Dashboard & Live Trace Viewer
tests/
  test_loop.py     # Agent loop, strikes, loop detection tests
  test_faults.py   # Fault injection & 503 recovery tests
  test_policy.py   # Risk gate & user approval tests
  test_verifier.py # Read-back verifier rejection tests
requirements.txt
README.md
```

---

## 3. Generalization Proof (3 Distinct Tasks, 0 Core Changes)

The unchanged `worker/core/` engine successfully executes three completely different tasks:

1. **Invoice to ERP Processing (`tasks/invoice.py`)**: Reads unread emails, parses invoice metadata from PDF attachments, posts to ERP, and verifies posting with direct ERP read-back.
2. **Expense Reconciliation (`tasks/expense_recon.py`)**: Parses CSV expense reports, queries ERP ledger entries, detects price mismatches or missing entries, and reports anomalies.
3. **Customer Record Update (`tasks/customer_update.py`)**: Searches customer database, updates contact fields with `irreversible` risk gate approval, and verifies updated fields via read-back.

---

## 4. How to Run

### Setup Virtual Environment
```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Run Pytest Test Suite
```bash
.\.venv\Scripts\python.exe -m pytest -v
```

### Run CLI Engine
```bash
# Run Task 1 (Invoice) with Scripted Model:
.\.venv\Scripts\python.exe ui/cli.py --task invoice --mode scripted

# Run Task 2 (Expense Recon):
.\.venv\Scripts\python.exe ui/cli.py --task expense --mode scripted

# Run Task 3 (Customer Update):
.\.venv\Scripts\python.exe ui/cli.py --task customer --mode scripted

# Run with OpenRouter API (requires API Key):
$env:OPENROUTER_API_KEY="sk-or-v1-..."
.\.venv\Scripts\python.exe ui/cli.py --task invoice --mode openrouter
```

### Run Web UI & Live Trace Viewer
```bash
.\.venv\Scripts\python.exe ui/web.py
```
Open **http://127.0.0.1:8000** in your browser to view the interactive dashboard, launch tasks, monitor step-by-step trace streams, and manage policy approvals.
