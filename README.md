# AI Evaluation Framework

An end-to-end evaluation harness for LLMs, RAG pipelines, Security Guardrails, and Autonomous AI Agents using **DeepEval**, **Google Gemini**, and judge models (**Ollama** locally or **Groq** in the cloud).

---

## 🏗️ Architecture & Capability Domains

The framework organizes evaluation tests into **4 capability domains** under `tests/` driven by pure ground-truth Golden Datasets in `data/`:

```
ai_evaluation/
├── data/                                 # Standardized Golden Datasets
│   ├── core/single_turn.json             # Core Q&A Golden Cases
│   ├── rag/rag_pipeline.json             # RAG Ground-Truth Contexts & Answers
│   ├── security/prompt_injection.json    # Security Injection Payloads
│   └── agents/agent_trajectory.json      # Agent Trajectory Expectations
│
├── src/ai_evaluation/                    # System Under Test (SUT) Implementations
│   ├── agents/minimal_agent.py           # Gemini Multi-Tool Sequential Agent SUT
│   ├── rag_pipeline.py                   # RAG Pipeline SUT
│   ├── rag_retriever.py                  # BM25 Retriever SUT
│   └── llm_generator.py                  # Core LLM Response Generator SUT
│
└── tests/                                # Evaluation Test Suites
    ├── core/                             # Domain 1: Core Q&A & Summarization
    │   ├── test_single_turn.py           # Correctness & Relevancy Quality
    │   └── test_summarization.py         # Summary Alignment & Truthfulness
    ├── rag/                              # Domain 2: RAG Pipeline
    │   └── test_rag_pipeline.py          # Faithfulness & Contextual Recall
    ├── security/                         # Domain 3: Security & Guardrails
    │   ├── test_security_evaluation.py   # Prompt Injection Safety
    │   └── test_safety.py                # Toxicity & Bias Guardrails
    └── agents/                           # Domain 4: Autonomous Agent Evaluation
        ├── test_tool_selection.py        # Single-Tool Selection Fit
        ├── test_tool_trajectory.py       # Sequential Ordering & Task Completion
        ├── test_tool_step_efficiency.py  # Step Efficiency & Redundancy
        ├── test_agent_loop_detection.py  # Real SUT Infinite Loop Detection
        └── *_metric_validation.py        # Controlled Evaluator Defect Suites
```

---

## 🔬 Evaluation Methodology

The framework uses a **Dual-Layer Evaluation Methodology**:

1. **Evaluator Validation Suites (`*_metric_validation.py`)**:
   - Controlled synthetic defect injection tests that verify the judge metric itself (e.g. proving `ToolCorrectnessMetric` flags wrong tools or out-of-order calls, `TaskCompletionMetric` flags incomplete tasks or hallucinations after tool errors, and `AgentLoopDetectionMetric` flags infinite retries).
2. **Real SUT Golden Dataset Suites**:
   - Live execution tests of the actual SUT (e.g. `MinimalAgent`) parameterized dynamically via `load_golden_cases` from `data/`.

---

## 📊 Agent Evaluation Matrix

```
                             Agent Evaluation
                                    │
       ┌────────────────────────────┴────────────────────────────┐
       │                                                         │
Process Evaluation                                       Outcome & Resilience
       │                                                         │
 ├── Tool Selection (ToolCorrectnessMetric)                ├── Task Completion (TaskCompletionMetric)
 ├── Trajectory Ordering (ToolCorrectnessMetric)           └── Loop & Error Recovery (AgentLoopDetectionMetric)
 └── Step Efficiency (StepEfficiencyMetric)
```

---

## 🚀 Setup & Requirements

### Requirements
- Python 3.10+
- A Google Gemini API key (`GEMINI_API_KEY`)
- Either [Ollama](https://ollama.com/) running locally or a [Groq](https://groq.com/) API key

### 1. Environment Setup
```bash
python -m venv deepeval_venv
source deepeval_venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration (`.env`)
Create a `.env` file in the project root:
```ini
EVALUATION_JUDGE=groq
GROQ_EVALUATION_MODEL=qwen/qwen3.8-27b
GEMINI_API_KEY="your-gemini-api-key"
GROQ_API_KEY="your-groq-api-key"

# Telemetry Opt-Out (Disables telemetry noise & shutdown delays)
DEEPEVAL_TELEMETRY_OPT_OUT=YES
POSTHOG_DISABLED=1
```

---

## 🧪 Running Evaluation Tests

### Run Domain-Specific Test Suites

```bash
# Domain 1: Core Quality & Summarization
./deepeval_venv/bin/pytest -v tests/core/

# Domain 2: RAG Pipeline
./deepeval_venv/bin/pytest -v tests/rag/

# Domain 3: Security & Safety Guardrails
./deepeval_venv/bin/pytest -v tests/security/

# Domain 4: Autonomous Agent Evaluation Suite
./deepeval_venv/bin/pytest -v tests/agents/
```

### Run Full Test Suite
```bash
./deepeval_venv/bin/pytest -v tests/
```

---

## 🛠️ Project Configuration & Fixtures

- `tests/conftest.py`: Shared judge fixtures (`ollama_judge_model`, `groq_judge_model`, `judge_model`), dataset loader (`load_golden_cases`), and telemetry setup.
- `pyproject.toml`: Pytest configuration (`testpaths = ["tests"]`, `norecursedirs = ["tests/static", "deepeval_venv", ".git"]`).
