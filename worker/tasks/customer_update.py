"""Task Config 3: Customer Record Update."""

from typing import Any, Dict, Optional
from worker.core.llm import FinalAnswer
from worker.core.loop import TaskConfig
from worker.core.registry import ToolRegistry
from worker.core.verifier import VerifierResult
from worker.env.mock_erp import MockERPEnvironment
from worker.tools.erp import create_erp_tools


def create_customer_update_task(
    erp_env: Optional[MockERPEnvironment] = None,
    customer_id: str = "CUST-101",
    target_field: str = "email",
    new_value: str = "apexpay@apextech.com",
) -> TaskConfig:
    erp = erp_env or MockERPEnvironment()
    tools = create_erp_tools(erp)

    def verify_customer_update(final_ans: FinalAnswer, registry: ToolRegistry) -> VerifierResult:
        """Independent verification read-back check."""
        cust = erp.get_customer(customer_id)
        if not cust:
            return VerifierResult(
                passed=False,
                reason=f"Read-back failed: Customer '{customer_id}' not found in ERP.",
            )

        actual_val = cust.get(target_field)
        if actual_val != new_value:
            return VerifierResult(
                passed=False,
                reason=f"Read-back failed: Field '{target_field}' is '{actual_val}', expected '{new_value}'.",
            )

        return VerifierResult(
            passed=True,
            reason=f"Read-back verified: Customer '{customer_id}' field '{target_field}' updated to '{new_value}'.",
            details={"customer": cust},
        )

    return TaskConfig(
        name="Customer Record Update",
        goal=f"Locate customer '{customer_id}', update field '{target_field}' to '{new_value}', and verify the change.",
        tools=tools,
        system_hint="Use search_customer or get_customer to verify customer existence. Use update_customer_record to modify field.",
        success_check=verify_customer_update,
        approval_rules={"require_approval_for_risk": ["irreversible"]},
        max_steps=10,
        initial_facts={
            "customer_id": customer_id,
            "target_field": target_field,
            "new_value": new_value,
        },
    )
