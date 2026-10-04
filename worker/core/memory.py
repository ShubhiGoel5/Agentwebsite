"""Memory store for holding facts, goal, history, and compact context construction."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StepRecord:
    step_num: int
    action_type: str  # "tool_call", "final", "ask", "invalid"
    action_data: Dict[str, Any]
    observation: Optional[str] = None
    timestamp: float = 0.0


class Memory:
    """Stores task facts, conversation history, and provides compact LLM context windows."""

    def __init__(self, goal: str, system_hint: Optional[str] = None, max_recent_steps: int = 6):
        self.goal = goal
        self.system_hint = system_hint
        self.max_recent_steps = max_recent_steps
        self.facts: Dict[str, Any] = {}
        self.history: List[StepRecord] = []
        self.user_answers: List[Dict[str, str]] = []

    def add_fact(self, key: str, value: Any) -> None:
        self.facts[key] = value

    def add_user_answer(self, question: str, answer: str) -> None:
        self.user_answers.append({"question": question, "answer": answer})

    def record_step(
        self,
        step_num: int,
        action_type: str,
        action_data: Dict[str, Any],
        observation: Optional[str] = None,
    ) -> StepRecord:
        record = StepRecord(
            step_num=step_num,
            action_type=action_type,
            action_data=action_data,
            observation=observation,
        )
        self.history.append(record)
        return record

    def build_context(self) -> List[Dict[str, Any]]:
        """Constructs a compact context window for the LLM."""
        messages: List[Dict[str, Any]] = []

        # System prompt hint
        sys_content = "You are an autonomous task execution engine. You reason step by step and execute tools to achieve the goal."
        if self.system_hint:
            sys_content += f"\nTask Domain Guidance:\n{self.system_hint}"
        
        sys_content += (
            "\n\nRules:"
            "\n1. Domain knowledge lives in tool schemas, not system assumptions."
            "\n2. Perform tools carefully. Verification will run after you finish."
            "\n3. If missing information required for a high-risk write, ask the user."
            "\n4. If a tool fails with an error, observe the error and adapt your parameters."
        )
        messages.append({"role": "system", "content": sys_content})

        # Goal + Facts prompt
        user_msg = f"GOAL: {self.goal}\n"
        if self.facts:
            user_msg += f"\nKNOWN FACTS / INITIAL STATE:\n{self.facts}\n"
        if self.user_answers:
            user_msg += "\nUSER CLARIFICATIONS:\n"
            for ua in self.user_answers:
                user_msg += f"- Question: {ua['question']}\n  Answer: {ua['answer']}\n"
        
        messages.append({"role": "user", "content": user_msg})

        # Compact history (sliding window of last N steps)
        recent_steps = self.history[-self.max_recent_steps :] if len(self.history) > self.max_recent_steps else self.history

        if len(self.history) > len(recent_steps):
            summary_msg = f"[Note: {len(self.history) - len(recent_steps)} earlier step(s) truncated for brevity]"
            messages.append({"role": "system", "content": summary_msg})

        for rec in recent_steps:
            if rec.action_type == "tool_call":
                tool_name = rec.action_data.get("name", "unknown")
                tool_args = rec.action_data.get("args", {})
                asst_content = f"Action: Tool Call -> {tool_name}({tool_args})"
                messages.append({"role": "assistant", "content": asst_content})
                if rec.observation is not None:
                    messages.append({"role": "user", "content": f"Observation for {tool_name}:\n{rec.observation}"})

            elif rec.action_type == "invalid":
                err = rec.action_data.get("error", "Invalid format")
                messages.append({"role": "assistant", "content": f"Action: [Invalid Tool Call format]"})
                messages.append({"role": "user", "content": f"System Feedback: {err}. Please retry with correct tool schema."})

            elif rec.action_type == "ask":
                q = rec.action_data.get("question", "")
                messages.append({"role": "assistant", "content": f"Action: Ask User -> {q}"})

            elif rec.action_type == "final":
                content = rec.action_data.get("content", "")
                messages.append({"role": "assistant", "content": f"Action: Final Claim -> {content}"})
                if rec.observation is not None:  # Verifier feedback
                    messages.append({"role": "user", "content": f"Verifier Feedback: {rec.observation}"})

        return messages
