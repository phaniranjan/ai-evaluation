"""LLM response generation backed by the Gemini API."""

import logging
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class LLMGenerator:
    """Generate single-turn and conversational responses with a Gemini model."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        max_output_tokens: int = 400,
    ) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_output_tokens = max_output_tokens

    def generate_response(self, question: str) -> str:
        """Generate a response to a standalone question."""
        logger.debug("Generating response using model: %s", self.model)
        logger.debug("Question: %s", question)
        return self._generate(question)

    def generate_conversation_turn(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_instruction: Optional[str] = None,
    ) -> str:
        """Generate an assistant response using the supplied conversation history."""
        logger.debug("Generating conversation turn using model: %s", self.model)
        logger.debug("User message: %s", user_message)
        if conversation_history:
            logger.debug("Conversation history: %d turns", len(conversation_history))

        messages = self._build_messages(conversation_history, user_message)
        return self._generate(messages, system_instruction)

    def _generate(self, contents: Any, system_instruction: Optional[str] = None) -> str:
        config = types.GenerateContentConfig(
            max_output_tokens=self.max_output_tokens,
            system_instruction=system_instruction,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        result = response.text.strip()
        logger.debug("Generated response (length: %d chars)", len(result))
        return result

    @staticmethod
    def _build_messages(
        conversation_history: Optional[List[Dict[str, str]]], user_message: str
    ) -> List[types.Content]:
        messages = []
        for turn in conversation_history or []:
            messages.append(
                types.Content(
                    role="user" if turn["role"] == "user" else "model",
                    parts=[types.Part(text=turn["content"])],
                )
            )
        messages.append(types.Content(role="user", parts=[types.Part(text=user_message)]))
        return messages
