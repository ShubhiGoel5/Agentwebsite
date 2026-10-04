"""Policy gate for checking action risk levels and managing user approvals."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union

from worker.core.registry import RiskLevel, Tool


class PolicyDecision(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_APPROVAL = "needs_approval"


@dataclass
class PolicyCheckResult:
    decision: PolicyDecision
    reason: str
    risk_level: RiskLevel
    tool_name: str
    tool_args: Dict[str, Any]


ApprovalCallback = Callable[[str, Dict[str, Any], RiskLevel, str], bool]


class PolicyGate:
    """Evaluates risk levels for tool calls and enforces approval rules."""

    def __init__(
        self,
        approval_rules: Optional[Dict[str, Any]] = None,
        approval_callback: Optional[ApprovalCallback] = None,
    ):
        """
        approval_rules format example:
        {
            "require_approval_for_risk": ["write", "irreversible"],
            "require_approval_for_tools": ["delete_customer", "erp_post_invoice"],
            "auto_approve": False
        }
        """
        self.approval_rules = approval_rules or {}
        self.approval_callback = approval_callback

    def check(self, tool: Tool, args: Dict[str, Any]) -> PolicyCheckResult:
        risk: RiskLevel = tool.risk
        tool_name = tool.name

        required_risks = self.approval_rules.get("require_approval_for_risk", ["irreversible"])
        required_tools = self.approval_rules.get("require_approval_for_tools", [])

        needs_approval = (risk in required_risks) or (tool_name in required_tools)

        if not needs_approval:
            return PolicyCheckResult(
                decision=PolicyDecision.APPROVED,
                reason=f"Action '{tool_name}' ({risk}) is within auto-approval policy.",
                risk_level=risk,
                tool_name=tool_name,
                tool_args=args,
            )

        # Action requires approval
        if self.approval_callback is not None:
            approved = self.approval_callback(
                tool_name, args, risk, f"Tool '{tool_name}' requires approval (Risk: {risk})."
            )
            if approved:
                return PolicyCheckResult(
                    decision=PolicyDecision.APPROVED,
                    reason=f"User approved execution of '{tool_name}'.",
                    risk_level=risk,
                    tool_name=tool_name,
                    tool_args=args,
                )
            else:
                return PolicyCheckResult(
                    decision=PolicyDecision.REJECTED,
                    reason=f"User explicitly rejected execution of '{tool_name}'.",
                    risk_level=risk,
                    tool_name=tool_name,
                    tool_args=args,
                )

        return PolicyCheckResult(
            decision=PolicyDecision.NEEDS_APPROVAL,
            reason=f"Tool '{tool_name}' with risk '{risk}' requires user approval.",
            risk_level=risk,
            tool_name=tool_name,
            tool_args=args,
        )
