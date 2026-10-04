"""Scripted LLM Adapter for deterministic testing without external API calls."""

from typing import Any, Dict, List, Optional, Union

from worker.core.llm import AskUser, FinalAnswer, LLMClient, LLMResponse, ToolCall


class ScriptedLLMAdapter:
    """Fake model adapter that yields pre-defined responses for test scenarios."""

    def __init__(self, script: List[Union[LLMResponse, ToolCall, FinalAnswer, AskUser, str]]):
        self.script: List[LLMResponse] = []
        for item in script:
            if isinstance(item, LLMResponse):
                self.script.append(item)
            elif isinstance(item, ToolCall):
                self.script.append(LLMResponse(action_type="tool_call", action=item))
            elif isinstance(item, FinalAnswer):
                self.script.append(LLMResponse(action_type="final", action=item))
            elif isinstance(item, AskUser):
                self.script.append(LLMResponse(action_type="ask", action=item))
            elif isinstance(item, str):
                self.script.append(LLMResponse(action_type="invalid", error_message=item))
            else:
                raise ValueError(f"Invalid script item type: {type(item)}")

        self.call_count = 0

    def decide(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        if self.call_count >= len(self.script):
            # Default fallback if script runs past end
            return LLMResponse(
                action_type="final",
                action=FinalAnswer(content="[Scripted LLM] End of script reached."),
            )

        resp = self.script[self.call_count]
        self.call_count += 1
        return resp
