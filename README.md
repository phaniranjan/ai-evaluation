# AI Evaluation Tests

A small DeepEval test suite that generates LLM responses with Google Gemini and
evaluates them with a separate judge model. Ollama is the default judge, and
Groq-hosted Llama is available as a cloud-based alternative. The repository
includes a single-turn correctness test and a multi-turn professionalism test.

## Requirements

- Python 3.10 or newer
- A Google Gemini API key
- Either [Ollama](https://ollama.com/) running locally, or a [Groq](https://groq.com/) API key for Llama

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv deepeval_venv
source deepeval_venv/bin/activate
pip install -r requirements.txt
```

Set the Gemini API key in the current shell:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

Download the local judge model and leave the Ollama service running:

```bash
ollama pull qwen2.5:7b
ollama serve
```

The judge defaults to `qwen2.5:7b`. Override it, or point tests at a remote
Ollama service, with:

```bash
export OLLAMA_EVALUATION_MODEL="qwen2.5:7b"
export OLLAMA_BASE_URL="http://localhost:11434"
```

### Use Groq Llama as the judge

Set the Groq API key and select Groq before running the tests:

```bash
export GROQ_API_KEY="your-groq-api-key"
export EVALUATION_JUDGE="groq"
export GROQ_EVALUATION_MODEL="qwen/qwen3.6-27b"
```

`EVALUATION_JUDGE` defaults to `ollama`, so no changes are required for the
existing local setup. Set it to `groq` to use Llama 3.3 70B on Groq instead.

You can also load variables from a local `.env` file:

```bash
set -a; source .env; set +a
```

Keep `.env` and `.env.local` out of Git. They are ignored by `.gitignore`.

## Run Tests

Run the single-turn response quality tests (evaluating correctness and answer relevancy):

```bash
./deepeval_venv/bin/pytest -q tests/static/test_single_turn.py
./deepeval_venv/bin/pytest -q tests/dynamic/test_single_turn_dynamic.py
```

Run the multi-turn conversational quality tests:

```bash
./deepeval_venv/bin/pytest -q tests/static/test_multi_turn.py
./deepeval_venv/bin/pytest -q tests/dynamic/test_multi_turn_dynamic.py
```

Run the summarization evaluation tests:

```bash
./deepeval_venv/bin/pytest -q tests/static/test_summarization.py
./deepeval_venv/bin/pytest -q tests/dynamic/test_summarization_dynamic.py
```

Run the safety and guardrails evaluation tests (bias and toxicity):

```bash
./deepeval_venv/bin/pytest -q tests/static/test_safety.py
./deepeval_venv/bin/pytest -q tests/dynamic/test_safety_dynamic.py
```

DeepEval does not allow `LLMTestCase` and `ConversationalTestCase` to be evaluated in the same test run, so run these modules as separate pytest commands.

## Project Files

- `conftest.py` - Shared judge-model fixture (Ollama or Groq); Gemini remains the response generator.
- `test_data/` - JSON data used by the evaluation tests.
- `test_single_turn.py` / `test_single_turn_dynamic.py` - Evaluates single-turn LLM responses for correctness (`GEval`) and answer relevancy (`AnswerRelevancyMetric`).
- `test_multi_turn.py` / `test_multi_turn_dynamic.py` - Evaluates multi-turn conversations using `ConversationalGEval`.
- `test_summarization.py` / `test_summarization_dynamic.py` - Evaluates document summaries for keypoint alignment and truthfulness using `SummarizationMetric`.
- `test_safety.py` / `test_safety_dynamic.py` - Stress-tests model safety against biased and toxic content using `BiasMetric` and `ToxicityMetric`.
- `requirements.txt` - Python dependencies.
- `.gitignore` - Excludes environment files, virtual environments, and test caches.
