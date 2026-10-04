"""Task Config 2: Expense Reconciliation Matching."""

from typing import Any, Dict, Optional
from worker.core.llm import FinalAnswer
from worker.core.loop import TaskConfig
from worker.core.registry import ToolRegistry
from worker.core.verifier import VerifierResult
from worker.env.mock_erp import MockERPEnvironment
from worker.tools.erp import create_erp_tools
from worker.tools.files import create_file_tools


SAMPLE_EXPENSE_CSV = """date,description,amount,category
2026-09-01,Software Subscription,150.00,Software
2026-09-05,Team Lunch,95.50,Meals
2026-09-12,Client Flight,420.00,Travel
2026-09-18,Office Stationeries,65.00,Office
2026-09-22,Taxi fare,35.00,Travel"""


def create_expense_recon_task(
    erp_env: Optional[MockERPEnvironment] = None,
    csv_data: str = SAMPLE_EXPENSE_CSV,
) -> TaskConfig:
    erp = erp_env or MockERPEnvironment()
    tools = create_erp_tools(erp) + create_file_tools()

    def verify_reconciliation(final_ans: FinalAnswer, registry: ToolRegistry) -> VerifierResult:
        data = final_ans.result_data or {}
        mismatches = data.get("mismatches") or []
        
        # We expect 2 anomalies: Team Lunch (95.50 vs ERP 85.50) and Taxi fare (35.00 missing in ERP)
        if not mismatches or len(mismatches) < 2:
            return VerifierResult(
                passed=False,
                reason="Verification failed: Reconciliation report did not flag expected discrepancies (Team Lunch amount mismatch & missing Taxi fare).",
            )
        return VerifierResult(
            passed=True,
            reason=f"Reconciliation verified: Identified {len(mismatches)} expense discrepancies correctly.",
            details={"mismatches": mismatches},
        )

    return TaskConfig(
        name="Expense Reconciliation",
        goal="Parse submitted expense CSV data, query ERP expense entries using get_erp_expenses, match entries, and report any amount mismatches or missing records.",
        tools=tools,
        system_hint="Use read_csv_file to parse input CSV, then get_erp_expenses to fetch ledger entries. Compare amounts and report discrepancies in result_data['mismatches'].",
        success_check=verify_reconciliation,
        approval_rules={"require_approval_for_risk": ["irreversible"]},
        max_steps=15,
        initial_facts={"csv_data": csv_data},
    )
