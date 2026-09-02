import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

from deepeval.models import GeminiModel, OllamaModel

from ai_evaluation.agents.minimal_agent import MinimalAgent
from ai_evaluation.groq_model import GroqModel
from ai_evaluation.llm_generator import LLMGenerator
from ai_evaluation.rag_pipeline import RAGPipeline
from ai_evaluation.rag_retriever import BM25Retriever


# Configure logging
def _configure_logging():
    """Configure logging for tests."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"test_execution_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


_configure_logging()
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def gemini_model():
    """Provide the Gemini model retained for generation and Gemini-specific tests."""
    logger.info("Initializing Gemini model")
    return GeminiModel(
        model="gemini-3.5-flash-lite",
        api_key=os.environ["GEMINI_API_KEY"],
    )


@pytest.fixture(scope="session")
def ollama_judge_model():
    """Provide the local LLM used exclusively as the DeepEval judge."""
    model = os.getenv("OLLAMA_EVALUATION_MODEL", "qwen2.5:7b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    logger.info("Initializing Ollama evaluation model: %s", model)
    return OllamaModel(model=model, base_url=base_url, temperature=0)


@pytest.fixture(scope="session")
def groq_judge_model():
    """Provide Groq-hosted model as the cloud-based DeepEval judge."""
    model = os.getenv("GROQ_EVALUATION_MODEL", "qwen/qwen3.6-27b")
    logger.info("Initializing Groq evaluation model: %s", model)
    return GroqModel(
        model=model,
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0,
    )


@pytest.fixture(scope="session")
def judge_model(request):
    """Provide the configured DeepEval judge (Ollama by default, or Groq)."""
    provider = os.getenv("EVALUATION_JUDGE", "ollama").lower()
    fixtures = {
        "ollama": "ollama_judge_model",
        "groq": "groq_judge_model",
    }
    try:
        return request.getfixturevalue(fixtures[provider])
    except KeyError as error:
        valid_providers = ", ".join(sorted(fixtures))
        raise pytest.UsageError(
            "EVALUATION_JUDGE must be one of: {}".format(valid_providers)
        ) from error


@pytest.fixture
def api_key():
    """Fixture for accessing the Gemini API key."""
    return os.environ["GEMINI_API_KEY"]


def load_golden_cases(
    filename: str, domain: str = "core", prefix_filter: Optional[str] = None
) -> List[dict]:
    """Load golden dataset test cases at collection time for pytest parameterization."""
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / domain / filename
    with data_path.open(encoding="utf-8") as f:
        cases = json.load(f)
    if prefix_filter:
        return [c for c in cases if c.get("id", "").startswith(prefix_filter)]
    return cases


@pytest.fixture
def test_data_loader() -> Callable[..., dict]:
    """Provide a loader for evaluation dataset JSON files located in data/<domain>/<filename>."""

    def load_test_data(filename: str, domain: str = "core") -> dict:
        project_root = Path(__file__).parent.parent
        data_path = project_root / "data" / domain / filename
        with data_path.open(encoding="utf-8") as file:
            return json.load(file)

    return load_test_data


@pytest.fixture
def llm_generator(api_key):
    """Provide a configured Gemini response generator."""
    return LLMGenerator(api_key)


@pytest.fixture
def minimal_agent(api_key):
    """Provide a configured MinimalAgent with single tool selection capability."""
    return MinimalAgent(api_key)


@pytest.fixture
def response_generator(llm_generator):
    """Fixture for generating LLM responses."""
    return llm_generator.generate_response


@pytest.fixture
def conversation_generator(llm_generator):
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
            assistant_response = llm_generator.generate_conversation_turn(
                user_msg,
                conversation_history=history[:-1],
                system_instruction=initial_prompt,
            )
            conversation.append({"role": "assistant", "content": assistant_response})
            history.append({"role": "assistant", "content": assistant_response})

        return conversation

    return generate_conversation


@pytest.fixture
def rag_retriever():
    """Fixture initializing BM25Retriever loaded with news articles."""
    data_path = (
        Path(__file__).parent.parent / "src" / "ai_evaluation" / "data" / "rag_news_articles.json"
    )
    with data_path.open(encoding="utf-8") as f:
        articles = json.load(f)

    retriever = BM25Retriever()
    retriever.add_articles(articles)
    return retriever


@pytest.fixture
def rag_pipeline(llm_generator, rag_retriever):
    """Fixture returning RAGPipeline backed by BM25Retriever and LLMGenerator."""
    return RAGPipeline(retriever=rag_retriever, llm_generator=llm_generator, default_top_k=2)
