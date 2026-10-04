"""OpenRouter LLM Adapter supporting model rotation fallback, 429 backoff, and tool calling."""

import json
import os
import time
from typing import Any, Dict, List, Optional

import requests

from worker.core.llm import AskUser, FinalAnswer, LLMClient, LLMResponse, ToolCall


import sys

DEFAULT_FALLBACK_MODELS = [
    "apodex/apodex-1.1-mini:free",
    "inclusionai/ling-3.0-flash-sante:free",
    "dots-studio/dots-3-note-preview:free",
    "liquid/lfm-2.5-2.6b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "cohere/north-mini-code:free",
]


class OpenRouterLLMAdapter:
    """OpenRouter OpenAI-compatible adapter with model rotation and budget controls."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_list: Optional[List[str]] = None,
        max_daily_budget: int = 50,
        site_url: str = "https://github.com/autonomous-task-worker",
        app_name: str = "Autonomous Task Worker",
    ):
        key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        if not key and sys.platform == "win32":
            try:
                import winreg
                reg_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment")
                val, _ = winreg.QueryValueEx(reg_key, "OPENROUTER_API_KEY")
                if val:
                    key = val
            except Exception:
                pass
        self.api_key = key
        self.model_list = model_list or DEFAULT_FALLBACK_MODELS
        self.max_daily_budget = max_daily_budget
        self.request_count = 0
        self.site_url = site_url
        self.app_name = app_name
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def _format_system_instructions(self, tools: List[Dict[str, Any]], custom_prompt: Optional[str] = None) -> str:
        instructions = (
            "You are an Autonomous AI Agent executing step-by-step actions using available tools."
            "\nYou MUST respond in strictly valid JSON matching one of the following formats:"
            "\n"
            "\nFormat 1 (To call a tool):"
            '\n{\n  "action_type": "tool_call",\n  "name": "<tool_name>",\n  "args": { ... }\n}'
            "\n"
            "\nFormat 2 (If you need to ask the user for required input):"
            '\n{\n  "action_type": "ask",\n  "question": "<question_text>",\n  "options": ["<opt1>", "<opt2>"]\n}'
            "\n"
            "\nFormat 3 (When task is fully completed):"
            '\n{\n  "action_type": "final",\n  "content": "<explanation>",\n  "result_data": { ... }\n}'
            "\n"
            "\nDO NOT include markdown code block formatting outside the raw JSON object if possible."
        )
        if custom_prompt:
            instructions += f"\n\nTask-Specific Prompt:\n{custom_prompt}"
        return instructions

    def decide(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        if self.request_count >= self.max_daily_budget:
            print(f"[Warning] Approaching or exceeding daily request budget ({self.request_count}/{self.max_daily_budget})")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
            "Content-Type": "application/json",
        }

        sys_inst = self._format_system_instructions(tools, system_prompt)
        formatted_msgs = [{"role": "system", "content": sys_inst}] + messages

        payload = {
            "messages": formatted_msgs,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        # Model rotation loop across fallback models
        last_exception = None
        for model in self.model_list:
            payload["model"] = model
            
            # Retry loop for 429/50x transient errors
            for attempt in range(3):
                if not self.api_key:
                    return LLMResponse(
                        action_type="invalid",
                        error_message="OPENROUTER_API_KEY environment variable is not set.",
                    )

                try:
                    self.request_count += 1
                    response = requests.post(self.endpoint, headers=headers, json=payload, timeout=30)

                    if response.status_code == 200:
                        data = response.json()
                        raw_content = data["choices"][0]["message"]["content"]
                        return self._parse_response(raw_content)

                    elif response.status_code == 429:
                        wait_sec = 2 ** (attempt + 1)
                        print(f"[OpenRouter 429 Rate Limit] Model {model}. Retrying in {wait_sec}s...")
                        time.sleep(wait_sec)
                        continue

                    elif response.status_code in (500, 502, 503, 504):
                        time.sleep(1)
                        continue

                    else:
                        err_text = f"API HTTP {response.status_code}: {response.text}"
                        print(f"[OpenRouter Error] {err_text}")
                        last_exception = Exception(err_text)
                        break  # Try next model

                except Exception as e:
                    last_exception = e
                    time.sleep(1)
                    continue

        return LLMResponse(
            action_type="invalid",
            error_message=f"All models in fallback list failed. Last error: {last_exception}",
        )

    def _parse_response(self, raw_str: str) -> LLMResponse:
        """Parses model output into structured LLMResponse."""
        cleaned = raw_str.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            return LLMResponse(
                action_type="invalid",
                raw_response=raw_str,
                error_message=f"JSON decoding error: {e.msg} in model output: '{raw_str[:100]}...'",
            )

        act_type = parsed.get("action_type") or parsed.get("type")

        if act_type == "tool_call" or "name" in parsed and "args" in parsed:
            name = parsed.get("name")
            args = parsed.get("args", {})
            if not name or not isinstance(args, dict):
                return LLMResponse(
                    action_type="invalid",
                    raw_response=raw_str,
                    error_message="Tool call must specify 'name' (str) and 'args' (dict).",
                )
            return LLMResponse(
                action_type="tool_call",
                action=ToolCall(name=name, args=args),
                raw_response=raw_str,
            )

        elif act_type == "ask" or "question" in parsed:
            q = parsed.get("question", "User input required.")
            opts = parsed.get("options", [])
            return LLMResponse(
                action_type="ask",
                action=AskUser(question=q, options=opts),
                raw_response=raw_str,
            )

        elif act_type == "final" or "content" in parsed:
            content = parsed.get("content", "Task completed.")
            res_data = parsed.get("result_data", {})
            return LLMResponse(
                action_type="final",
                action=FinalAnswer(content=content, result_data=res_data),
                raw_response=raw_str,
            )

        return LLMResponse(
            action_type="invalid",
            raw_response=raw_str,
            error_message=f"Unrecognized response structure keys: {list(parsed.keys())}",
        )
