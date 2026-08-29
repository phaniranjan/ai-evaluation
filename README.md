# AI Evaluation Tests

A small DeepEval test suite that evaluates LLM responses with Google Gemini. The repository includes a single-turn correctness test and a multi-turn professionalism test.

## Requirements

- Python 3.10 or newer
- A Google Gemini API key

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

You can also load variables from a local `.env` file:

```bash
set -a; source .env; set +a
```

Keep `.env` and `.env.local` out of Git. They are ignored by `.gitignore`.

## Run Tests

Run the single-turn correctness test:

```bash
./deepeval_venv/bin/pytest -q test_correctness.py
```

Run the conversational professionalism test:

```bash
./deepeval_venv/bin/pytest -q test_professionalism.py
```

DeepEval does not allow `LLMTestCase` and `ConversationalTestCase` to be evaluated in the same test run, so run these modules as separate pytest commands.

## Project Files

- `conftest.py` - Shared session-scoped Gemini model fixture.
- `test_correctness.py` - Evaluates a single LLM response against an expected response using `GEval`.
- `test_professionalism.py` - Evaluates a multi-turn conversation using `ConversationalGEval`.
- `requirements.txt` - Python dependencies.
- `.gitignore` - Excludes environment files, virtual environments, and test caches.
