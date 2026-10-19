"""Bind each provider first, then compose LangChain's native fallback runnable."""

from collections.abc import Sequence
from typing import Any

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .config import Settings


class ModelFactory:
    """Preserve callbacks, config, async, batch, and streaming fallback behavior."""

    def __init__(self, settings: Settings):
        settings.require_chat()
        common = {"temperature": 0, "timeout": settings.timeout, "max_retries": 0}
        self.models = []
        if settings.deepseek_api_key:
            self.models.append(ChatOpenAI(
                api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url,
                model=settings.deepseek_model, **common,
            ))
        if settings.qwen_api_key:
            self.models.append(ChatOpenAI(
                api_key=settings.qwen_api_key, base_url=settings.qwen_base_url,
                model=settings.qwen_model, **common,
            ))

    @staticmethod
    def _combine(models: Sequence[Runnable]) -> Runnable:
        """A single configured provider works without a dummy fallback credential."""
        return models[0].with_fallbacks(list(models[1:])) if len(models) > 1 else models[0]

    def chat(self) -> Runnable:
        return self._combine(self.models)

    def tools(self, tools: list[Any]) -> Runnable:
        # Do not send OpenAI-specific parallel_tool_calls to compatible providers.
        # The workflow also validates that the response contains only one tool call.
        return self._combine([model.bind_tools(tools) for model in self.models])

    def structured(self, schema: type[BaseModel]) -> Runnable:
        # ChatOpenAI >=0.3 defaults to json_schema, which failed in the user's run.
        return self._combine([
            model.with_structured_output(schema, method="function_calling")
            for model in self.models
        ])
