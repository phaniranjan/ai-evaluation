"""System Under Test (SUT) evaluation testing MinimalAgent tool selection loaded from Golden Dataset."""

import logging

import pytest
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

logger = logging.getLogger(__name__)


@pytest.mark.dynamic
def test_minimal_agent_tool_selection_from_golden(judge_model, minimal_agent, test_data_loader):
    """SUT Agent Test: Evaluate MinimalAgent tool selection using test case loaded from Golden Dataset."""
    # 1. Load golden test case from data/agents/agent_trajectory.json
    cases = test_data_loader("agent_trajectory.json", domain="agents")
    case = cases[0]  # agent_tool_selection_01

    user_query = case["input"]
    expected_tools_data = case["expected_tools"]

    logger.info("Executing Golden Test Case [%s]: %s", case["id"], case["name"])
    logger.info("Input Query: %s", user_query)

    # 2. Execute MinimalAgent SUT
    result = minimal_agent.run(user_query)
    actual_tools_called = result["tools_called"]
    actual_answer = result["answer"]

    # 3. Convert expected_tools from Golden JSON to DeepEval ToolCall instances
    expected_tools = [
        ToolCall(
            name=tool["name"],
            input_parameters=tool.get("input_parameters"),
        )
        for tool in expected_tools_data
    ]

    # 4. Construct LLMTestCase
    test_case = LLMTestCase(
        input=user_query,
        actual_output=actual_answer,
        tools_called=actual_tools_called,
        expected_tools=expected_tools,
    )

    # 5. Measure using DeepEval ToolCorrectnessMetric
    metric = ToolCorrectnessMetric(threshold=0.7, model=judge_model)
    metric.measure(test_case)

    logger.info(
        "ToolCorrectnessMetric score for Golden Case [%s]: %.2f (Reason: %s)",
        case["id"],
        metric.score,
        metric.reason,
    )

    assert (
        metric.score >= 0.7
    ), f"Expected ToolCorrectnessMetric >= 0.7 for '{user_query}', but got score: {metric.score}"

    logger.info("Golden Tool Selection Test Case [%s] passed successfully!", case["id"])
