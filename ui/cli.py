"""CLI interface for running Autonomous Task Worker tasks interactively."""

import argparse
import os
import sys
from pathlib import Path

# Ensure worker package is in sys.path
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
from worker.tasks.scheduling import create_scheduling_task
from worker.env.mock_calendar import MockCalendarEnvironment


def cli_approval_callback(tool_name: str, args: dict, risk: str, reason: str) -> bool:
    print("\n" + "=" * 60)
    print(f"⚠️  POLICY APPROVAL REQUIRED [{risk.upper()} RISK]")
    print(f"Tool: {tool_name}")
    print(f"Arguments: {args}")
    print(f"Reason: {reason}")
    print("=" * 60)
    choice = input("Do you approve this execution? (y/N): ").strip().lower()
    return choice in ("y", "yes")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Autonomous Task Worker CLI Engine")
    parser.add_argument(
        "--task",
        choices=["invoice", "expense", "customer", "scheduling"],
        default="invoice",
        help="Task configuration to execute (default: invoice)",
    )
    parser.add_argument(
        "--mode",
        choices=["scripted", "openrouter"],
        default="scripted",
        help="LLM Adapter mode to use (default: scripted)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENROUTER_API_KEY", ""),
        help="OpenRouter API Key (if mode=openrouter)",
    )
    parser.add_argument(
        "--trace-out",
        default="trace_log.jsonl",
        help="Path for trace log output (default: trace_log.jsonl)",
    )

    args = parser.parse_args()

    # Shared environments
    mail_env = MockMailEnvironment()
    erp_env = MockERPEnvironment()
    calendar_env = MockCalendarEnvironment()

    # Select Task Config
    if args.task == "invoice":
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
                content="Invoice #INV-2026-9912 for $4,250.00 processed from email msg_001 and successfully posted to ERP.",
                result_data={"invoice_number": "INV-2026-9912", "amount": 4250.0},
            ),
        ]
    elif args.task == "expense":
        task_config = create_expense_recon_task(erp_env=erp_env)
        default_script = [
            ToolCall(name="read_csv_file", args={"csv_content": task_config.initial_facts["csv_data"]}),
            ToolCall(name="get_erp_expenses", args={}),
            FinalAnswer(
                content="Reconciliation complete. Found 2 mismatches.",
                result_data={
                    "mismatches": [
                        {"description": "Team Lunch", "csv_amount": 95.50, "erp_amount": 85.50},
                        {"description": "Taxi fare", "csv_amount": 35.00, "erp_amount": None, "note": "Missing in ERP"},
                    ]
                },
            ),
        ]
    elif args.task == "customer":
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
    else:  # scheduling
        task_config = create_scheduling_task(calendar_env=calendar_env)
        default_script = [
            ToolCall(name="contacts.find", args={"name": "Priya"}),
            ToolCall(name="contacts.find", args={"name": "Marcus Vance"}),
            ToolCall(
                name="calendar.free_busy",
                args={"emails": ["priya.sharma@acme.com", "marcus.vance@acme.com"],
                      "start_date": "2026-10-12", "end_date": "2026-10-16"},
            ),
            ToolCall(
                name="calendar.hold",
                args={"slot_start": "2026-10-13T10:00:00",
                      "slot_end": "2026-10-13T10:45:00",
                      "attendees": ["priya.sharma@acme.com", "marcus.vance@acme.com"]},
            ),
            ToolCall(
                name="calendar.create_event",
                args={"slot_start": "2026-10-13T10:00:00",
                      "slot_end": "2026-10-13T10:45:00",
                      "attendees": ["priya.sharma@acme.com", "marcus.vance@acme.com"],
                      "title": "Q4 Budget Review",
                      "duration_min": 45},
            ),
            FinalAnswer(
                content="Meeting scheduled on 2026-10-13 at 10:00 AM for 45 mins with Priya and Marcus.",
                result_data={"event_id": "evt_101"},
            ),
        ]

    # Select LLM Client Adapter
    if args.mode == "openrouter":
        if not args.api_key:
            print("❌ Error: OPENROUTER_API_KEY is required when --mode=openrouter.")
            sys.exit(1)
        llm_client = OpenRouterLLMAdapter(api_key=args.api_key)
    else:
        llm_client = ScriptedLLMAdapter(script=default_script)

    policy_gate = PolicyGate(
        approval_rules=task_config.approval_rules,
        approval_callback=cli_approval_callback,
    )
    trace_logger = TraceLogger(log_path=args.trace_out)

    print(f"\n🚀 Starting Task: {task_config.name}")
    print(f"🎯 Goal: {task_config.goal}")
    print(f"🤖 Adapter Mode: {args.mode.upper()}")
    print("-" * 60)

    loop = AgentLoop(
        config=task_config,
        llm_client=llm_client,
        policy_gate=policy_gate,
        trace_logger=trace_logger,
    )

    res = loop.run()

    print("-" * 60)
    print(f"📊 Task Execution Status: {res.status.upper()}")
    print(f"⏱️ Steps Taken: {res.steps_taken}")
    print(f"📝 Reason: {res.reason}")

    if res.final_answer:
        print(f"\n✅ Final Answer Content:\n{res.final_answer.content}")
        if res.final_answer.result_data:
            print(f"📌 Result Data: {res.final_answer.result_data}")

    print(f"\n📂 Trace Log written to: {args.trace_out}")


if __name__ == "__main__":
    main()
