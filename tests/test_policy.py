"""Unit tests for policy gate risk checking and user approval workflows."""

from worker.adapters.scripted import ScriptedLLMAdapter
from worker.core.llm import FinalAnswer, ToolCall
from worker.core.loop import AgentLoop, TaskConfig
from worker.core.policy import PolicyDecision, PolicyGate
from worker.env.mock_erp import MockERPEnvironment
from worker.tools.erp import create_erp_tools


def test_policy_gate_auto_approve_read_and_pause_irreversible():
    erp_env = MockERPEnvironment()
    tools = create_erp_tools(erp_env)

    # update_customer_record is tagged as risk='irreversible'
    script = [
        ToolCall(
            name="update_customer_record",
            args={
                "customer_id": "CUST-101",
                "field_name": "email",
                "new_value": "newemail@apex.com",
            },
        )
    ]

    adapter = ScriptedLLMAdapter(script)

    config = TaskConfig(
        name="Policy Test",
        goal="Update customer email",
        tools=tools,
        approval_rules={"require_approval_for_risk": ["irreversible"]},
    )

    loop = AgentLoop(config=config, llm_client=adapter)
    res = loop.run()

    # Without approval callback, loop pauses for approval
    assert res.status == "paused_for_user"
    assert res.ask_user is not None


def test_policy_gate_approval_callback_rejection():
    erp_env = MockERPEnvironment()
    tools = create_erp_tools(erp_env)

    # Callback rejects approval
    def reject_callback(tool_name, args, risk, msg):
        return False

    policy_gate = PolicyGate(
        approval_rules={"require_approval_for_risk": ["irreversible"]},
        approval_callback=reject_callback,
    )

    # Model tries irreversible tool, gets rejected, then makes final answer acknowledging rejection
    script = [
        ToolCall(
            name="update_customer_record",
            args={
                "customer_id": "CUST-101",
                "field_name": "email",
                "new_value": "newemail@apex.com",
            },
        ),
        FinalAnswer(content="Action was rejected by policy."),
    ]

    adapter = ScriptedLLMAdapter(script)

    config = TaskConfig(
        name="Policy Rejection Test",
        goal="Update customer email",
        tools=tools,
    )

    loop = AgentLoop(config=config, llm_client=adapter, policy_gate=policy_gate)
    res = loop.run()

    assert res.status == "success"
    # Verify customer email was NOT updated because callback rejected it
    cust = erp_env.get_customer("CUST-101")
    assert cust["email"] == "billing@apextech.com"
