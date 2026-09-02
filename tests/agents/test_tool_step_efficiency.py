"""System Under Test (SUT) evaluation testing MinimalAgent step efficiency loaded from Golden Dataset."""

import logging

import pytest
from deepeval.metrics import StepEfficiencyMetric
from deepeval.test_case import LLMTestCase

from tests.conftest import load_golden_cases

logger = logging.getLogger(__name__)

# Load golden test cases for pytest parameterization
_trajectory_cases = load_golden_cases(
    "agent_trajectory.json", domain="agents", prefix_filter="agent_trajectory"
)


@pytest.mark.parametrize("case", _trajectory_cases, ids=[c["id"] for c in _trajectory_cases])
@pytest.mark.dynamic
def test_minimal_agent_step_efficiency(judge_model, minimal_agent, case):
    """SUT Agent Test: Evaluate MinimalAgent step efficiency against Golden Dataset expectations."""
    query = case["input"]

    logger.info("Executing Golden Step Efficiency Test Case [%s]: %s", case["id"], case["name"])
    logger.info("Input Query: %s", query)

    # 1. Execute MinimalAgent SUT with sequential execution
    result = minimal_agent.run_with_execution(query)
    execution_log = result["execution_log"]
    final_answer = result["answer"]

    # 2. Convert execution_log to DeepEval _trace_dict tree structure
    children = [
        {
            "name": entry["tool_name"],
            "type": "tool",
            "input": {"inputParameters": entry["args"]},
            "output": entry["result"],
            "children": [],
        }
        for entry in execution_log
    ]

    trace_dict = {
        "name": "MinimalAgent",
        "type": "agent",
        "input": {"input": query},
        "children": children,
    }

    # 3. Construct LLMTestCase
    test_case = LLMTestCase(
        input=query,
        actual_output=final_answer,
    )
    test_case._trace_dict = trace_dict

    # 4. Measure using DeepEval StepEfficiencyMetric
    metric = StepEfficiencyMetric(threshold=0.7, model=judge_model)
    metric.measure(test_case)

    logger.info(
        "StepEfficiencyMetric score for Golden Case [%s]: %.2f (Reason: %s)",
        case["id"],
        metric.score,
        metric.reason,
    )

    assert (
        metric.score >= 0.7
    ), f"Expected StepEfficiencyMetric score >= 0.7 for [{case['id']}], but got: {metric.score}"

    logger.info("Golden Step Efficiency Test Case [%s] passed successfully!", case["id"])
