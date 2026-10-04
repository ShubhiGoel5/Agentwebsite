"""Domain-agnostic Agent Execution Loop enforcing reliability, policy checks, loop detection, and verifiers."""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from worker.core.llm import AskUser, FinalAnswer, LLMClient, LLMResponse, ToolCall
from worker.core.memory import Memory
from worker.core.policy import PolicyDecision, PolicyGate
from worker.core.registry import ToolRegistry, ToolResult
from worker.core.trace import TraceLogger
from worker.core.verifier import Verifier, VerifierResult


@dataclass
class LoopResult:
    status: str  # "success", "failed", "paused_for_user", "max_steps_exceeded", "escalated"
    final_answer: Optional[FinalAnswer] = None
    ask_user: Optional[AskUser] = None
    reason: str = ""
    steps_taken: int = 0
    trace_file: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskConfig:
    name: str
    goal: str
    tools: List[Any]  # List of Tool objects
    system_hint: Optional[str] = None
    success_check: Optional[Any] = None
    approval_rules: Optional[Dict[str, Any]] = None
    max_steps: int = 25
    initial_facts: Optional[Dict[str, Any]] = None


class AgentLoop:
    """Core autonomous task execution loop."""

    def __init__(
        self,
        config: TaskConfig,
        llm_client: LLMClient,
        policy_gate: Optional[PolicyGate] = None,
        verifier: Optional[Verifier] = None,
        trace_logger: Optional[TraceLogger] = None,
        max_strikes: int = 3,
        max_repeated_calls: int = 3,
    ):
        self.config = config
        self.llm_client = llm_client
        self.max_strikes = max_strikes
        self.max_repeated_calls = max_repeated_calls

        # Build registry from task tools
        self.registry = ToolRegistry()
        for tool in config.tools:
            self.registry.register(tool)

        # Build policy gate and verifier
        self.policy_gate = policy_gate or PolicyGate(approval_rules=config.approval_rules)
        self.verifier = verifier or Verifier(success_check=config.success_check)
        self.trace = trace_logger or TraceLogger()

        # Build memory store
        self.memory = Memory(goal=config.goal, system_hint=config.system_hint)
        if config.initial_facts:
            for k, v in config.initial_facts.items():
                self.memory.add_fact(k, v)

        # Loop counters & call tracking
        self.step = 0
        self.strikes = 0
        self.call_history: List[str] = []

    def _detect_loop(self, tool_name: str, tool_args: Dict[str, Any]) -> bool:
        """Checks if exact same tool call with exact same args was made max_repeated_calls times consecutively."""
        sig = f"{tool_name}:{sorted(tool_args.items())}"
        self.call_history.append(sig)

        if len(self.call_history) >= self.max_repeated_calls:
            recent = self.call_history[-self.max_repeated_calls :]
            if len(set(recent)) == 1:
                return True
        return False

    def run(self) -> LoopResult:
        """Executes the agent loop until task completion, pause, or step limit."""
        self.trace.log("task_start", self.step, f"Starting task '{self.config.name}'", {"goal": self.config.goal})

        while self.step < self.config.max_steps:
            self.step += 1

            # 1. Build context window
            ctx_messages = self.memory.build_context()
            openai_tools = self.registry.to_openai_schemas()

            self.trace.log("llm_request", self.step, "Querying LLM client", {"message_count": len(ctx_messages)})

            # 2. Query LLM Client
            try:
                response: LLMResponse = self.llm_client.decide(
                    messages=ctx_messages,
                    tools=openai_tools,
                    system_prompt=self.config.system_hint,
                )
            except Exception as e:
                self.trace.log("llm_error", self.step, f"LLM client error: {e}")
                return LoopResult(
                    status="failed",
                    reason=f"LLM Client exception: {str(e)}",
                    steps_taken=self.step,
                    trace_file=str(self.trace.log_path) if self.trace.log_path else None,
                )

            # 3. Handle Invalid / Malformed response
            if response.is_invalid():
                self.strikes += 1
                err_msg = response.error_message or "Malformed response format from model."
                self.trace.log("strike_count", self.step, f"Strike {self.strikes}/{self.max_strikes}: {err_msg}")
                self.memory.record_step(self.step, "invalid", {"error": err_msg})

                if self.strikes >= self.max_strikes:
                    self.trace.log("escalated", self.step, "Exceeded maximum strikes for invalid model responses.")
                    return LoopResult(
                        status="escalated",
                        reason=f"Model failed to generate valid action schema after {self.max_strikes} consecutive attempts. Error: {err_msg}",
                        steps_taken=self.step,
                    )
                continue

            # Reset strikes on valid action
            self.strikes = 0

            # 4. Handle Ask User action
            if response.is_ask():
                ask_act: AskUser = response.action  # type: ignore
                self.trace.log("ask_user", self.step, f"Model asked user: {ask_act.question}", ask_act.to_dict())
                self.memory.record_step(self.step, "ask", ask_act.to_dict())
                return LoopResult(
                    status="paused_for_user",
                    ask_user=ask_act,
                    reason="Model requested information or user clarification.",
                    steps_taken=self.step,
                )

            # 5. Handle Final Answer Claim
            if response.is_final():
                final_act: FinalAnswer = response.action  # type: ignore
                self.trace.log("final_claim", self.step, "Model claimed task completion.", final_act.to_dict())

                # Verifier Check
                v_res: VerifierResult = self.verifier.verify(final_act, self.registry)
                self.trace.log(
                    "verification_check",
                    self.step,
                    f"Verifier result: {'Passed' if v_res.passed else 'Failed'}",
                    {"passed": v_res.passed, "reason": v_res.reason, "details": v_res.details},
                )

                if v_res.passed:
                    self.memory.record_step(self.step, "final", final_act.to_dict(), observation="VERIFIED")
                    self.trace.log("task_end", self.step, "Task successfully completed and verified.")
                    return LoopResult(
                        status="success",
                        final_answer=final_act,
                        reason=f"Task completed successfully. Verification: {v_res.reason}",
                        steps_taken=self.step,
                    )
                else:
                    # Feed verification failure back into memory as observation
                    feedback = f"Verification Failed: {v_res.reason}. Please fix the output state or tools before claiming completion."
                    self.memory.record_step(self.step, "final", final_act.to_dict(), observation=feedback)
                    continue

            # 6. Handle Tool Call Action
            if response.is_tool_call():
                tool_call: ToolCall = response.action  # type: ignore
                tool = self.registry.get(tool_call.name)

                self.trace.log(
                    "tool_call_decided",
                    self.step,
                    f"Model decided tool call: '{tool_call.name}'",
                    tool_call.to_dict(),
                )

                if not tool:
                    err = f"Tool '{tool_call.name}' is not registered."
                    self.memory.record_step(self.step, "invalid", {"error": err})
                    continue

                # Loop detection
                if self._detect_loop(tool_call.name, tool_call.args):
                    msg = f"Loop detected: tool '{tool_call.name}' with args {tool_call.args} was called {self.max_repeated_calls} consecutive times."
                    self.trace.log("loop_detected", self.step, msg)
                    return LoopResult(
                        status="escalated",
                        reason=f"Execution paused due to repetitive loop detection. {msg}",
                        steps_taken=self.step,
                    )

                # Policy gate check
                p_check = self.policy_gate.check(tool, tool_call.args)
                self.trace.log(
                    "policy_check",
                    self.step,
                    f"Policy check for '{tool.name}': {p_check.decision.value}",
                    {"decision": p_check.decision.value, "risk": p_check.risk_level, "reason": p_check.reason},
                )

                if p_check.decision == PolicyDecision.REJECTED:
                    obs = f"[Policy Rejection] Execution of '{tool.name}' was rejected by policy: {p_check.reason}"
                    self.memory.record_step(self.step, "tool_call", tool_call.to_dict(), observation=obs)
                    continue
                elif p_check.decision == PolicyDecision.NEEDS_APPROVAL:
                    ask_act = AskUser(
                        question=f"Action '{tool.name}' requires approval (Risk level: {tool.risk}). Approve execution?",
                        options=["Approve", "Reject"],
                    )
                    return LoopResult(
                        status="paused_for_user",
                        ask_user=ask_act,
                        reason=f"Action '{tool.name}' requires user approval.",
                        steps_taken=self.step,
                    )

                # Execute Tool with Harness Transient Retry
                res: ToolResult = self.registry.execute(tool_call.name, tool_call.args)
                
                # Harness level transient retry (503 / timeout backoff)
                retry_count = 0
                while not res.success and res.is_transient and retry_count < 2:
                    retry_count += 1
                    time.sleep(0.5 * retry_count)
                    self.trace.log("transient_retry", self.step, f"Retrying transient failure in '{tool.name}' (Attempt {retry_count+1})")
                    res = self.registry.execute(tool_call.name, tool_call.args)

                obs_str = res.to_observation_str()
                self.trace.log(
                    "tool_execution",
                    self.step,
                    f"Executed tool '{tool.name}'. Success: {res.success}",
                    {"success": res.success, "error": res.error, "output": res.output},
                )

                self.memory.record_step(self.step, "tool_call", tool_call.to_dict(), observation=obs_str)

        self.trace.log("max_steps_exceeded", self.step, f"Task reached max step limit ({self.config.max_steps}).")
        return LoopResult(
            status="max_steps_exceeded",
            reason=f"Agent loop reached maximum configured limit of {self.config.max_steps} steps without completion.",
            steps_taken=self.step,
        )
