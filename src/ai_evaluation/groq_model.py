"""Groq-hosted model compatible with DeepEval metrics."""

import asyncio
import logging
import os
import time
from typing import Optional, Tuple, Type, Union

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.models.llms.utils import trim_and_load_json
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class GroqModel(DeepEvalBaseLLM):
    """Wrap a Groq-hosted model so DeepEval can use it as an evaluator."""

    def __init__(
        self,
        model: str = "qwen/qwen3.6-27b",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ["GROQ_API_KEY"]
        super().__init__(model)

    def load_model(self):
        try:
            from groq import Groq
        except ImportError as error:
            raise ImportError(
                "groq is required to use GroqModel. "
                "Install it with: pip install groq"
            ) from error
        return Groq(api_key=self._api_key, max_retries=5)

    def generate(
        self, prompt: str, schema: Optional[Type[BaseModel]] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        client = self.load_model()
        request = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if schema is not None:
            request["response_format"] = {"type": "json_object"}

        from groq import APIStatusError, RateLimitError

        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                response = client.chat.completions.create(**request)
                break
            except (RateLimitError, APIStatusError) as exc:
                is_429 = getattr(exc, "status_code", None) == 429 or isinstance(exc, RateLimitError)
                if not is_429 or attempt == max_attempts - 1:
                    raise
                wait_seconds = (2 ** attempt) + 1
                logger.warning(
                    "Groq rate limit hit (429). Retrying in %d seconds (attempt %d/%d)...",
                    wait_seconds,
                    attempt + 1,
                    max_attempts,
                )
                time.sleep(wait_seconds)

        output = response.choices[0].message.content or ""
        if schema is not None:
            return schema.model_validate(trim_and_load_json(output)), 0.0
        return output, 0.0

    async def a_generate(
        self, prompt: str, schema: Optional[Type[BaseModel]] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.generate(prompt, schema))

    def get_model_name(self) -> str:
        return f"{self.model_name} (Groq)"
