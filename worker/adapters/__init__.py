"""Worker Adapters Package."""

from worker.adapters.openrouter import OpenRouterLLMAdapter
from worker.adapters.scripted import ScriptedLLMAdapter

__all__ = ["ScriptedLLMAdapter", "OpenRouterLLMAdapter"]
