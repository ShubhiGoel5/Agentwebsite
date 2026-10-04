"""Unit tests for fault injection, transient 503 recovery, and parameter repair."""

from worker.adapters.scripted import ScriptedLLMAdapter
from worker.core.llm import FinalAnswer, ToolCall
from worker.core.loop import AgentLoop, TaskConfig
from worker.env.mock_erp import MockERPEnvironment
from worker.tools.erp import create_erp_tools


def test_transient_503_recovery_via_harness_retry():
    erp_env = MockERPEnvironment()
    # Inject 1 transient error
    erp_env.transient_error_counter = 1

    tools = create_erp_tools(erp_env)

    script = [
        ToolCall(name="get_customer", args={"customer_id": "CUST-101"}),
        FinalAnswer(content="Customer found.", result_data={"customer_id": "CUST-101"}),
    ]

    adapter = ScriptedLLMAdapter(script)

    config = TaskConfig(
        name="Fault Injection Test",
        goal="Fetch customer despite 503 fault",
        tools=tools,
    )

    loop = AgentLoop(config=config, llm_client=adapter)
    res = loop.run()

    # Harness should retry automatically and succeed
    assert res.status == "success"


def test_schema_error_fed_back_to_model_and_repaired():
    erp_env = MockERPEnvironment()
    tools = create_erp_tools(erp_env)

    # Step 1: Tool call missing required 'vendor_id' parameter
    # Step 2: Repaired Tool Call with required parameters
    # Step 3: Final Answer
    script = [
        ToolCall(
            name="post_erp_invoice",
            args={"invoice_number": "INV-FAIL-01", "amount": 100.0},
        ),
        ToolCall(
            name="post_erp_invoice",
            args={
                "invoice_number": "INV-FAIL-01",
                "vendor_id": "VEND-01",
                "amount": 100.0,
            },
        ),
        FinalAnswer(content="Invoice posted after repair.", result_data={"invoice_number": "INV-FAIL-01"}),
    ]

    adapter = ScriptedLLMAdapter(script)

    config = TaskConfig(
        name="Schema Error Repair Test",
        goal="Test invalid schema feedback and repair",
        tools=tools,
    )

    loop = AgentLoop(config=config, llm_client=adapter)
    res = loop.run()

    assert res.status == "success"
    # Verify invoice was posted on the second attempt
    assert erp_env.get_invoice("INV-FAIL-01") is not None
