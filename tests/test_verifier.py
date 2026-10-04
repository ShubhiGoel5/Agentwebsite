"""Unit tests for independent verifier read-back checks and rejection handling."""

from worker.adapters.scripted import ScriptedLLMAdapter
from worker.core.llm import FinalAnswer, ToolCall
from worker.core.loop import AgentLoop, TaskConfig
from worker.core.verifier import VerifierResult
from worker.env.mock_erp import MockERPEnvironment
from worker.tools.erp import create_erp_tools


def test_verifier_rejects_premature_final_claim():
    erp_env = MockERPEnvironment()
    tools = create_erp_tools(erp_env)

    # Independent verifier check function
    def verifier_check(final_ans, registry):
        inv = erp_env.get_invoice("INV-VERIFY-99")
        if not inv:
            return VerifierResult(
                passed=False,
                reason="Invoice 'INV-VERIFY-99' missing from ERP ledger.",
            )
        return VerifierResult(passed=True, reason="Invoice verified in ERP.")

    # Script:
    # 1. Premature Final claim (rejected by verifier)
    # 2. Tool call posting invoice
    # 3. Valid Final claim (passed by verifier)
    script = [
        FinalAnswer(content="I claim invoice INV-VERIFY-99 is posted!"),
        ToolCall(
            name="post_erp_invoice",
            args={
                "invoice_number": "INV-VERIFY-99",
                "vendor_id": "VEND-01",
                "amount": 500.0,
            },
        ),
        FinalAnswer(content="Invoice posted now."),
    ]

    adapter = ScriptedLLMAdapter(script)

    config = TaskConfig(
        name="Verifier Test",
        goal="Post and verify invoice INV-VERIFY-99",
        tools=tools,
        success_check=verifier_check,
    )

    loop = AgentLoop(config=config, llm_client=adapter)
    res = loop.run()

    assert res.status == "success"
    # Ensure step count is 3 (attempt 1 failed verifier, tool called, attempt 3 passed)
    assert res.steps_taken == 3
    assert erp_env.get_invoice("INV-VERIFY-99") is not None
