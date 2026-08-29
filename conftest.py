import json
import os
from pathlib import Path
from typing import Callable

import pytest

from deepeval.models import GeminiModel


@pytest.fixture(scope="session")
def gemini_model():
    return GeminiModel(
        model="gemini-3.5-flash-lite",
        api_key=os.environ["GEMINI_API_KEY"],
    )


@pytest.fixture
def test_data_loader() -> Callable[[str], dict]:
    def load_test_data(filename: str) -> dict:
        data_path = Path(__file__).parent / "test_data" / filename
        with data_path.open(encoding="utf-8") as file:
            return json.load(file)

    return load_test_data