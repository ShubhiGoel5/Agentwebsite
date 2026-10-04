"""Independent verifier for success criteria checking and state read-backs."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union

from worker.core.llm import FinalAnswer
from worker.core.registry import ToolRegistry


@dataclass
class VerifierResult:
    passed: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


SuccessCheckCallable = Callable[[FinalAnswer, ToolRegistry], VerifierResult]


class Verifier:
    """Independent verification engine to validate task completion claims."""

    def __init__(self, success_check: Optional[Union[SuccessCheckCallable, Dict[str, Any]]] = None):
        self.success_check = success_check

    def verify(self, final_answer: FinalAnswer, registry: ToolRegistry) -> VerifierResult:
        """Runs verification checks against the claimed final answer and target environment."""
        if not self.success_check:
            # Default fallback verification: checks if content is non-empty
            if final_answer.content and len(final_answer.content.strip()) > 0:
                return VerifierResult(
                    passed=True,
                    reason="Default verification passed: final answer content provided.",
                )
            return VerifierResult(
                passed=False,
                reason="Default verification failed: final answer content was empty.",
            )

        # 1. Callable verification rule (e.g. read-back function)
        if callable(self.success_check):
            try:
                res = self.success_check(final_answer, registry)
                if isinstance(res, VerifierResult):
                    return res
                elif isinstance(res, tuple) and len(res) == 2:
                    return VerifierResult(passed=bool(res[0]), reason=str(res[1]))
                elif isinstance(res, bool):
                    return VerifierResult(
                        passed=res,
                        reason="Verification function returned " + ("success" if res else "failure"),
                    )
            except Exception as e:
                return VerifierResult(
                    passed=False,
                    reason=f"Verifier execution encountered an error: {type(e).__name__} - {str(e)}",
                )

        # 2. Schema / dict key matching rule
        if isinstance(self.success_check, dict):
            required_keys = self.success_check.get("required_result_keys", [])
            data = final_answer.result_data or {}
            missing = [k for k in required_keys if k not in data]
            if missing:
                return VerifierResult(
                    passed=False,
                    reason=f"Final result data is missing required key(s): {missing}",
                    details={"missing": missing, "provided_data": data},
                )
            return VerifierResult(
                passed=True,
                reason="All required result keys present in final answer.",
                details={"provided_data": data},
            )

        return VerifierResult(passed=False, reason="Invalid success_check type configured.")
