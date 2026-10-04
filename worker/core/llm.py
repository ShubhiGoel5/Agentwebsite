"""LLM Client Protocol and core action types."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, Union


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]
    call_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "args": self.args, "call_id": self.call_id}


@dataclass
class FinalAnswer:
    content: str
    summary: str = ""
    result_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "summary": self.summary,
            "result_data": self.result_data,
        }


@dataclass
class AskUser:
    question: str
    options: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"question": self.question, "options": self.options}


ActionType = Union[ToolCall, FinalAnswer, AskUser]


@dataclass
class LLMResponse:
    action_type: Literal["tool_call", "final", "ask", "invalid"]
    action: Optional[ActionType] = None
    raw_response: str = ""
    error_message: Optional[str] = None

    def is_tool_call(self) -> bool:
        return self.action_type == "tool_call" and isinstance(self.action, ToolCall)

    def is_final(self) -> bool:
        return self.action_type == "final" and isinstance(self.action, FinalAnswer)

    def is_ask(self) -> bool:
        return self.action_type == "ask" and isinstance(self.action, AskUser)

    def is_invalid(self) -> bool:
        return self.action_type == "invalid"


class LLMClient(Protocol):
    """Protocol interface for language model decision adapters."""

    def decide(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        ...
