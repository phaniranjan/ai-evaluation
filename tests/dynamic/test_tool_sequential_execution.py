"""System Under Test (SUT) evaluation testing MinimalAgent multi-tool sequential execution with DeepEval ToolCorrectnessMetric."""

import logging

import pytest
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams

logger = logging.getLogger(__name__)


@pytest.mark.dynamic
def test_minimal_agent_sequential_tool_execution(judge_model, minimal_agent):
    """SUT Agent Test: Evaluate MinimalAgent multi-tool sequential execution trajectory using DeepEval ToolCorrectnessMetric."""
    query = "What is the weather in Tokyo and convert the temperature to Fahrenheit."

    logger.info("Executing SUT sequential tool execution test for query: %s", query)
    result = minimal_agent.run_with_execution(query)

    execution_log = result["execution_log"]
    final_answer = result["answer"]

    # Log the complete captured sequence as requested
    logger.info("--- Captured Sequential Tool Execution Trajectory ---")
    for entry in execution_log:
        logger.info(
            "Step #%d | Tool: %s | Args: %s | Result: %s",
            entry["order"],
            entry["tool_name"],
            entry["args"],
            entry["result"],
        )
    logger.info("Final Agent Answer:\n%s", final_answer)

    # Convert returned execution_log into DeepEval ToolCall objects
    actual_trajectory = [
        ToolCall(
            name=entry["tool_name"],
            input_parameters=entry["args"],
            output=entry["result"],
        )
        for entry in execution_log
    ]

    # Create expected two-step trajectory dynamically capturing calculated expression format
    calc_expression = actual_trajectory[1].input_parameters.get("expression", "25 * 9 / 5 + 32")
    expected_trajectory = [
        ToolCall(
            name="get_weather",
            input_parameters={"location": "Tokyo"},
            output="The current temperature in Tokyo is 25°C with sunny skies.",
        ),
        ToolCall(
            name="calculator",
            input_parameters={"expression": calc_expression},
            output="77.0",
        ),
    ]

    # Build LLMTestCase
    test_case = LLMTestCase(
        input=query,
        actual_output=final_answer,
        tools_called=actual_trajectory,
        expected_tools=expected_trajectory,
    )

    # Instantiate built-in ToolCorrectnessMetric with trajectory evaluation parameters
    metric = ToolCorrectnessMetric(
        available_tools=[
            ToolCall(name="get_weather"),
            ToolCall(name="calculator"),
            ToolCall(name="get_time"),
        ],
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS, ToolCallParams.OUTPUT],
        should_consider_ordering=True,
        threshold=0.7,
        model=judge_model,
    )

    metric.measure(test_case)

    logger.info(
        "ToolCorrectnessMetric trajectory score: %.2f (Reason: %s)",
        metric.score,
        metric.reason,
    )

    assert (
        metric.score >= 0.7
    ), f"Expected ToolCorrectnessMetric trajectory score >= 0.7, but got: {metric.score}"

    logger.info(
        "DeepEval ToolCorrectnessMetric sequential trajectory evaluation passed successfully!"
    )
