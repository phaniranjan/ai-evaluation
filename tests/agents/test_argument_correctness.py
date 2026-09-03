"""System Under Test (SUT) evaluation testing MinimalAgent tool argument correctness loaded from Golden Dataset."""

import logging

import pytest
from deepeval.metrics import ArgumentCorrectnessMetric
from deepeval.test_case import LLMTestCase

from tests.conftest import load_golden_cases

logger = logging.getLogger(__name__)

# Load golden argument correctness test cases for pytest parameterization
_argument_cases = load_golden_cases(
    "agent_trajectory.json", domain="agents", prefix_filter="agent_argument"
)


@pytest.mark.parametrize("case", _argument_cases, ids=[c["id"] for c in _argument_cases])
@pytest.mark.dynamic
def test_minimal_agent_argument_correctness(minimal_agent, judge_model, case):
    """SUT Agent Test: Execute MinimalAgent against Golden Dataset argument correctness scenarios.

    DeepEval ArgumentCorrectnessMetric Limitation Note:
    DeepEval evaluates tool call arguments strictly against test_case.input without inspecting
    intermediate tool execution outputs (tool.output). For chained multi-tool trajectories (Case 13)
    where tool #2 parameters depend on tool #1's dynamic output, the judge scores 0.50.
    Case 13 specifies min_score = 0.5 to account for chained trajectory argument dependency.
    """
    query = case["input"]
    min_score = case.get("min_score", 0.7)

    logger.info(
        "Executing Golden Argument Correctness Test Case [%s]: %s", case["id"], case["name"]
    )
    logger.info("Input Query: %s | Expected Min Score: %.2f", query, min_score)

    result = minimal_agent.run_with_execution(query)

    test_case = LLMTestCase(
        input=query,
        actual_output=result["answer"],
        tools_called=result["tools_called"],
    )

    metric = ArgumentCorrectnessMetric(threshold=0.5, model=judge_model)
    metric.measure(test_case)

    logger.info(
        "ArgumentCorrectnessMetric score for Golden Case [%s]: %.2f (Reason: %s)",
        case["id"],
        metric.score,
        metric.reason,
    )

    assert (
        metric.score >= min_score
    ), f"Expected ArgumentCorrectnessMetric score >= {min_score} for Golden Case [{case['id']}], but got: {metric.score}"

    logger.info("Golden Argument Correctness Test Case [%s] passed successfully!", case["id"])
