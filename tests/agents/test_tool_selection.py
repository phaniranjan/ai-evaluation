"""System Under Test (SUT) evaluation testing MinimalAgent tool selection using Golden Dataset cases."""

import logging

import pytest
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from tests.conftest import load_golden_cases

logger = logging.getLogger(__name__)

# Load golden test cases for pytest parameterization
_selection_cases = load_golden_cases(
    "agent_trajectory.json", domain="agents", prefix_filter="agent_tool_selection"
)


@pytest.mark.parametrize("case", _selection_cases, ids=[c["id"] for c in _selection_cases])
@pytest.mark.dynamic
def test_minimal_agent_tool_selection(judge_model, minimal_agent, case):
    """SUT Agent Test: Evaluate MinimalAgent tool selection against Golden Dataset expectations."""
    user_query = case["input"]
    expected_tools_data = case["expected_tools"]

    logger.info("Executing Golden Test Case [%s]: %s", case["id"], case["name"])
    logger.info("Input Query: %s", user_query)

    # 1. Execute MinimalAgent SUT
    result = minimal_agent.run(user_query)
    actual_tools_called = result["tools_called"]
    actual_answer = result["answer"]

    # 2. Convert expected_tools from Golden JSON to DeepEval ToolCall instances
    expected_tools = [
        ToolCall(
            name=tool["name"],
            input_parameters=tool.get("input_parameters"),
        )
        for tool in expected_tools_data
    ]

    # 3. Construct LLMTestCase
    test_case = LLMTestCase(
        input=user_query,
        actual_output=actual_answer,
        tools_called=actual_tools_called,
        expected_tools=expected_tools,
    )

    # 4. Measure using DeepEval ToolCorrectnessMetric with SUT declared available_tools
    metric = ToolCorrectnessMetric(
        available_tools=minimal_agent.available_tools,
        threshold=0.7,
        model=judge_model,
    )
    metric.measure(test_case)

    logger.info(
        "ToolCorrectnessMetric score for Golden Case [%s]: %.2f (Reason: %s)",
        case["id"],
        metric.score,
        metric.reason,
    )

    assert (
        metric.score >= 0.7
    ), f"Expected ToolCorrectnessMetric >= 0.7 for [{case['id']}], but got score: {metric.score}"

    logger.info("Golden Tool Selection Test Case [%s] passed successfully!", case["id"])
