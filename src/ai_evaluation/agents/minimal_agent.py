"""Minimal one-tool agent demonstrating Gemini tool selection without execution or side effects."""

import logging
from typing import Any, Dict, List, Optional

from deepeval.test_case import ToolCall
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


def get_weather(location: str) -> str:
    """Get the current weather for a specified location."""
    return f"Weather information for {location}"


class MinimalAgent:
    """A minimal single-tool agent for evaluating Gemini tool selection capability."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
    ) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def run(self, user_prompt: str) -> Dict[str, Any]:
        """Send prompt to Gemini with single tool declaration and capture tool selection.

        Args:
            user_prompt: Input prompt from user.

        Returns:
            Dict containing 'answer' string and 'tools_called' list of DeepEval ToolCall instances.
        """
        logger.info("Executing MinimalAgent prompt: %s", user_prompt)
        config = types.GenerateContentConfig(
            tools=[get_weather],
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=config,
        )

        tools_called: List[ToolCall] = []
        if response.function_calls:
            for call in response.function_calls:
                logger.info("Gemini selected tool call: %s", call.name)
                tools_called.append(ToolCall(name=call.name))

        answer = response.text.strip() if response.text else ""

        return {
            "prompt": user_prompt,
            "answer": answer,
            "tools_called": tools_called,
        }
