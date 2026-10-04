"""Task Config 1: Invoice Processing & ERP Posting."""

from typing import Any, Dict, Optional
from worker.core.llm import FinalAnswer
from worker.core.loop import TaskConfig
from worker.core.registry import ToolRegistry
from worker.core.verifier import VerifierResult
from worker.env.mock_erp import MockERPEnvironment
from worker.env.mock_mail import MockMailEnvironment
from worker.tools.erp import create_erp_tools
from worker.tools.files import create_file_tools
from worker.tools.mail import create_mail_tools


def create_invoice_task(
    mail_env: Optional[MockMailEnvironment] = None,
    erp_env: Optional[MockERPEnvironment] = None,
    target_email_id: str = "msg_001",
) -> TaskConfig:
    mail = mail_env or MockMailEnvironment()
    erp = erp_env or MockERPEnvironment()

    tools = create_mail_tools(mail) + create_erp_tools(erp) + create_file_tools()

    def verify_invoice_posted(final_ans: FinalAnswer, registry: ToolRegistry) -> VerifierResult:
        """Independent verification read-back: checks ERP state directly."""
        data = final_ans.result_data or {}
        inv_num = data.get("invoice_number") or "INV-2026-9912"
        
        # Read-back query into ERP
        erp_record = erp.get_invoice(inv_num)
        if not erp_record:
            return VerifierResult(
                passed=False,
                reason=f"Read-back check failed: Invoice '{inv_num}' was not found in ERP ledger.",
            )

        if erp_record.get("amount") != 4250.0:
            return VerifierResult(
                passed=False,
                reason=f"Read-back check failed: Invoice amount in ERP is {erp_record.get('amount')}, expected 4250.0",
            )

        return VerifierResult(
            passed=True,
            reason=f"Read-back verified: Invoice '{inv_num}' exists in ERP with amount {erp_record.get('amount')}.",
            details={"erp_record": erp_record},
        )

    return TaskConfig(
        name="Invoice to ERP Processing",
        goal=f"Check unread emails for invoice message '{target_email_id}', extract invoice details from the PDF attachment, post the invoice into ERP, and verify posting.",
        tools=tools,
        system_hint="Retrieve email content using read_email. Extract invoice_number, vendor_id, and amount from PDF text. Post to ERP using post_erp_invoice.",
        success_check=verify_invoice_posted,
        approval_rules={"require_approval_for_risk": ["irreversible"]},
        max_steps=15,
        initial_facts={"target_email_id": target_email_id},
    )
