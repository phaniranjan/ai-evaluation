# AI Evaluation Framework

An end-to-end evaluation harness for LLMs, RAG pipelines, Security Guardrails, and Autonomous AI Agents using **DeepEval**, **Google Gemini**, and judge models (**Groq** in the cloud or **Ollama** locally).

---

## 🏗️ Architecture & Capability Domains

The framework organizes evaluation tests into **4 capability domains** under `tests/` driven by pure ground-truth Golden Datasets in `data/`:

```
ai_evaluation/
├── data/                                      # Standardized Golden Datasets
│   ├── core/single_turn.json                  # Core Q&A Golden Cases
│   ├── rag/rag_pipeline.json                  # RAG Ground-Truth Contexts & Answers
│   ├── security/prompt_injection.json         # Security Injection Payloads
│   └── agents/agent_evaluation.json           # Master Agent Evaluation Golden Cases
│
├── src/ai_evaluation/                         # System Under Test (SUT) Implementations
│   ├── agents/minimal_agent.py                # Gemini Multi-Tool Sequential Agent SUT (RBAC & Retry support)
│   ├── rag_pipeline.py                        # RAG Pipeline SUT
│   ├── rag_retriever.py                       # BM25 Retriever SUT
│   └── llm_generator.py                       # Core LLM Response Generator SUT
│
└── tests/                                     # Evaluation Test Suites
    ├── core/                                  # Domain 1: Core Q&A & Summarization
    │   ├── test_single_turn.py                # Correctness & Relevancy Quality
    │   └── test_summarization.py              # Summary Alignment & Truthfulness
    ├── rag/                                   # Domain 2: RAG Pipeline
    │   └── test_rag_pipeline.py               # Faithfulness & Contextual Relevancy
    ├── security/                              # Domain 3: Security & Guardrails
    │   ├── test_security_evaluation.py        # Prompt Injection Safety
    │   └── test_safety.py                     # Toxicity & Bias Guardrails
    └── agents/                                # Domain 4: Autonomous Agent Evaluation
        ├── test_tool_selection.py             # Single-Tool Selection Fit
        ├── test_tool_trajectory.py            # Sequential Ordering & Task Completion
        ├── test_tool_step_efficiency.py       # Step Efficiency & Redundancy
        ├── test_agent_loop_detection.py       # Real SUT Infinite Loop Detection
        ├── test_tool_permission.py            # Role-Based Tool Access (GUEST, USER, ADMIN)
        ├── test_argument_correctness.py       # Tool Argument Parameter Correctness
        └── *_metric_validation.py             # Controlled Evaluator Defect Validation Suites
```

---

## 🔬 Dual-Layer Evaluation Methodology

The framework uses a rigorous **Dual-Layer Evaluation Methodology**:

1. **Evaluator Validation Suites (`*_metric_validation.py`)**:
   - Controlled synthetic defect injection tests that verify the evaluator metric itself (e.g., proving `ToolPermissionMetric` flags unauthorized tool calls with `0.00`, `AgentLoopDetectionMetric` catches 10-turn retry loops with `0.00`, and `ArgumentCorrectnessMetric` flags wrong argument values with `0.00`).
2. **Real SUT Golden Dataset Suites**:
   - Live execution tests of the actual System Under Test (e.g., `MinimalAgent`) parameterized dynamically via `load_golden_cases` from `data/agents/agent_evaluation.json`.

---

## 📐 Evaluator Metrics Inventory

| Evaluation Domain | Metric Class | Purpose & Scope | Cost & Latency |
| :--- | :--- | :--- | :--- |
| **Core Q&A** | `AnswerRelevancyMetric` | Measures how relevantly the generated response addresses user prompt intent. | LLM Judge |
| **RAG** | `FaithfulnessMetric` | Evaluates if the answer is grounded strictly in retrieved context. | LLM Judge |
| **RAG** | `ContextualRelevancyMetric` | Measures precision of retrieved context chunks relative to input query. | LLM Judge |
| **Security** | `PromptInjectionMetric` | Evaluates resistance to indirect and direct prompt injection attacks. | LLM Judge |
| **Agents** | `ToolSelectionMetric` | Evaluates if the agent selected the appropriate tool for single-step queries. | Deterministic ($0 / 0ms) |
| **Agents** | `ToolTrajectoryMetric` | Evaluates sequential tool ordering and full task trajectory completion. | LLM Judge / Set Check |
| **Agents** | `ToolStepEfficiencyMetric` | Detects redundant tool calls and inefficient execution paths. | Deterministic ($0 / 0ms) |
| **Agents** | `AgentLoopDetectionMetric` | Flags infinite tool retry loops, repetitive calls, and failure to break out. | Deterministic ($0 / 0ms) |
| **Agents** | `ToolPermissionMetric` | Enforces Role-Based Access Control (RBAC) boundaries for `GUEST`, `USER`, and `ADMIN` roles. | Deterministic ($0 / 0ms) |
| **Agents** | `ArgumentCorrectnessMetric` | Verifies that tool input parameter values accurately match prompt intent. | LLM Judge |

---

## 📊 Agent Evaluation Matrix

```
                                      Agent Evaluation
                                             │
      ┌──────────────────────────────────────┼──────────────────────────────────────┐
      │                                      │                                      │
Process Evaluation                  Outcome & Resilience                   Security & Governance
      │                                      │                                      │
 ├── Tool Selection                   ├── Task Completion                    └── Tool Permission Boundaries
 │   (ToolSelectionMetric)            │   (ToolTrajectoryMetric)                     (ToolPermissionMetric - RBAC)
 ├── Trajectory Ordering              └── Loop & Error Recovery
 │   (ToolTrajectoryMetric)               (AgentLoopDetectionMetric)
 ├── Step Efficiency
 │   (ToolStepEfficiencyMetric)
 └── Argument Correctness
     (ArgumentCorrectnessMetric)
```

---

## 🚀 Setup & Requirements

### Prerequisites
- Python 3.10+
- A Google Gemini API key (`GEMINI_API_KEY`)
- Either a [Groq](https://groq.com/) API key (recommended) or [Ollama](https://ollama.com/) running locally.

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

## 🌿 Branch & Release Strategy

- **`main`**: Primary active development branch where new feature work and metric additions land.
- **`release/v0.1.0`**: Protected release maintenance branch preserving the stable `v0.1.0` framework release line.
- **`v0.1.0` (Tag)**: Immutable release checkpoint tag anchored to the `v0.1.0` release commit.

---

## 🛠️ Project Configuration & Fixtures

- `tests/conftest.py`: Shared judge fixtures (`ollama_judge_model`, `groq_judge_model`, `judge_model`), dataset loader (`load_golden_cases`), and telemetry setup.
- `pyproject.toml`: Pytest configuration (`testpaths = ["tests"]`, `norecursedirs = ["tests/static", "deepeval_venv", ".git"]`).
