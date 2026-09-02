"""System Under Test (SUT) evaluation testing MinimalAgent step efficiency with DeepEval StepEfficiencyMetric."""

import logging

import pytest
from deepeval.metrics import StepEfficiencyMetric
from deepeval.test_case import LLMTestCase

logger = logging.getLogger(__name__)


@pytest.mark.dynamic
def test_minimal_agent_step_efficiency(judge_model, minimal_agent):
    """SUT Agent Test: Evaluate MinimalAgent step efficiency trajectory using DeepEval StepEfficiencyMetric."""
    query = "What is the weather in Tokyo and convert the temperature to Fahrenheit."

    logger.info("Executing SUT step efficiency test for query: %s", query)
    result = minimal_agent.run_with_execution(query)

    execution_log = result["execution_log"]
    final_answer = result["answer"]

    # Convert execution_log to DeepEval _trace_dict tree structure
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

    test_case = LLMTestCase(
        input=query,
        actual_output=final_answer,
    )
    test_case._trace_dict = trace_dict

    metric = StepEfficiencyMetric(threshold=0.7, model=judge_model)
    metric.measure(test_case)

    logger.info(
        "StepEfficiencyMetric score on MinimalAgent trajectory: %.2f (Reason: %s)",
        metric.score,
        metric.reason,
    )

    assert (
        metric.score >= 0.7
    ), f"Expected StepEfficiencyMetric score >= 0.7, but got: {metric.score}"

    logger.info("MinimalAgent step efficiency evaluation passed successfully!")
