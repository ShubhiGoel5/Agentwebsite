"""Unit tests for agent execution loop, malformed JSON recovery, strikes, and loop detection."""

import pytest
from worker.adapters.scripted import ScriptedLLMAdapter
from worker.core.llm import AskUser, FinalAnswer, LLMResponse, ToolCall
from worker.core.loop import AgentLoop, TaskConfig
from worker.core.verifier import VerifierResult
from worker.env.mock_erp import MockERPEnvironment
from worker.env.mock_mail import MockMailEnvironment
from worker.tools.erp import create_erp_tools
from worker.tools.mail import create_mail_tools


def test_loop_success_with_scripted_model():
    mail_env = MockMailEnvironment()
    erp_env = MockERPEnvironment()

    tools = create_mail_tools(mail_env) + create_erp_tools(erp_env)

    # Scripted sequence: 1. read_email, 2. post_erp_invoice, 3. final claim
    script = [
        ToolCall(name="read_email", args={"email_id": "msg_001"}),
        ToolCall(
            name="post_erp_invoice",
            args={
                "invoice_number": "INV-2026-9912",
                "vendor_id": "VEND-ACME-01",
                "amount": 4250.0,
            },
        ),
        FinalAnswer(
            content="Invoice INV-2026-9912 posted to ERP.",
            result_data={"invoice_number": "INV-2026-9912"},
        ),
    ]

    adapter = ScriptedLLMAdapter(script)

    config = TaskConfig(
        name="Test Invoice Processing",
        goal="Post invoice from msg_001",
        tools=tools,
        max_steps=10,
    )

    loop = AgentLoop(config=config, llm_client=adapter)
    res = loop.run()

    assert res.status == "success"
    assert res.final_answer is not None
    assert erp_env.get_invoice("INV-2026-9912") is not None


def test_malformed_json_retry_and_strikes_escalation():
    mail_env = MockMailEnvironment()
    tools = create_mail_tools(mail_env)

    # 3 consecutive invalid responses
    script = [
        "Malformed string response 1",
        "Malformed string response 2",
        "Malformed string response 3",
    ]

    adapter = ScriptedLLMAdapter(script)

    config = TaskConfig(
        name="Test Strikes",
        goal="Test strike threshold",
        tools=tools,
        max_steps=5,
    )

    loop = AgentLoop(config=config, llm_client=adapter, max_strikes=3)
    res = loop.run()

    assert res.status == "escalated"
    assert "consecutive attempts" in res.reason.lower()


def test_loop_detection_escalation():
    erp_env = MockERPEnvironment()
    tools = create_erp_tools(erp_env)

    # Model calls exact same tool with exact same arguments 3 times consecutively
    script = [
        ToolCall(name="get_customer", args={"customer_id": "CUST-101"}),
        ToolCall(name="get_customer", args={"customer_id": "CUST-101"}),
        ToolCall(name="get_customer", args={"customer_id": "CUST-101"}),
    ]

    adapter = ScriptedLLMAdapter(script)

    config = TaskConfig(
        name="Test Loop Detect",
        goal="Test infinite loop prevention",
        tools=tools,
        max_steps=10,
    )

    loop = AgentLoop(config=config, llm_client=adapter, max_repeated_calls=3)
    res = loop.run()

    assert res.status == "escalated"
    assert "loop detected" in res.reason.lower()


def test_ask_user_pauses_execution():
    erp_env = MockERPEnvironment()
    tools = create_erp_tools(erp_env)

    script = [
        AskUser(question="Please specify vendor tax ID?", options=["Submit Tax ID", "Cancel"]),
    ]

    adapter = ScriptedLLMAdapter(script)

    config = TaskConfig(
        name="Test Ask User",
        goal="Test user query",
        tools=tools,
    )

    loop = AgentLoop(config=config, llm_client=adapter)
    res = loop.run()

    assert res.status == "paused_for_user"
    assert res.ask_user is not None
    assert "vendor tax ID" in res.ask_user.question
