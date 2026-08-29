import json
import logging
import os
from pathlib import Path
from typing import Callable

import pytest

from deepeval.models import GeminiModel
from src.llm_generator import generate_response, generate_conversation_turn


# Configure logging
def _configure_logging():
    """Configure logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("test_execution.log"),
        ],
    )


_configure_logging()
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def gemini_model():
    logger.info("Initializing Gemini model")
    return GeminiModel(
        model="gemini-3.5-flash-lite",
        api_key=os.environ["GEMINI_API_KEY"],
    )


@pytest.fixture
def api_key():
    """Fixture for accessing the Gemini API key."""
    return os.environ["GEMINI_API_KEY"]


@pytest.fixture
def test_data_loader() -> Callable[[str], dict]:
    def load_test_data(filename: str) -> dict:
        # Determine if this is dynamic or static data based on filename
        data_type = "dynamic" if "dynamic" in filename else "static"
        data_path = (
            Path(__file__).parent
            / "tests"
            / "data"
            / data_type
            / filename
        )
        with data_path.open(encoding="utf-8") as file:
            return json.load(file)

    return load_test_data


@pytest.fixture
def response_generator(api_key):
    """Fixture for generating LLM responses."""
    return lambda question: generate_response(question, api_key)


@pytest.fixture
def conversation_generator(api_key):
    """Fixture for generating multi-turn conversations."""
    def generate_conversation(initial_prompt: str, turns: list) -> list:
        """
        Generate a multi-turn conversation.
        
        Args:
            initial_prompt: Initial prompt or context for the conversation
            turns: List of user messages to generate responses for
        
        Returns:
            List of {"role": "user"/"assistant", "content": "..."} dicts
        """
        conversation = []
        history = []
        
        for turn_idx, user_msg in enumerate(turns):
            # Add user message
            conversation.append({"role": "user", "content": user_msg})
            history.append({"role": "user", "content": user_msg})
            
            # Generate assistant response
            assistant_response = generate_conversation_turn(user_msg, api_key, conversation_history=history)
            conversation.append({"role": "assistant", "content": assistant_response})
            history.append({"role": "assistant", "content": assistant_response})
        
        return conversation
    
    return generate_conversation