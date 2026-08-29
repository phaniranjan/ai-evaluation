import os

import pytest

from deepeval.models import GeminiModel


@pytest.fixture(scope="session")
def gemini_model():
    return GeminiModel(
        model="gemini-3.5-flash-lite",
        api_key=os.environ["GEMINI_API_KEY"],
    )