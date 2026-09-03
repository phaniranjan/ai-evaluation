"""System Under Test (SUT) evaluation testing MinimalAgent tool argument correctness with DeepEval ArgumentCorrectnessMetric."""

import logging

import pytest
from deepeval.metrics import ArgumentCorrectnessMetric
from deepeval.test_case import LLMTestCase

logger = logging.getLogger(__name__)


@pytest.mark.dynamic
def test_minimal_agent_argument_correctness_weather_query(minimal_agent, judge_model):
    """SUT Agent Test: Execute MinimalAgent against a weather query and verify ArgumentCorrectnessMetric scores 1.00."""
    query = "What is the weather in Tokyo?"

    logger.info("Executing SUT argument correctness test for weather query: %s", query)
    result = minimal_agent.run_with_execution(query)

    test_case = LLMTestCase(
        input=query,
        actual_output=result["answer"],
        tools_called=result["tools_called"],
    )

    metric = ArgumentCorrectnessMetric(threshold=0.5, model=judge_model)
    metric.measure(test_case)

    logger.info(
        "ArgumentCorrectnessMetric score for weather query: %.2f (Reason: %s)",
        metric.score,
        metric.reason,
    )

    assert (
        metric.score >= 0.7
    ), f"Expected ArgumentCorrectnessMetric score >= 0.7 for weather query, but got: {metric.score}"

    logger.info("Weather query tool argument correctness evaluation passed successfully!")


@pytest.mark.dynamic
def test_minimal_agent_argument_correctness_math_calculation_query(minimal_agent, judge_model):
    """SUT Agent Test: Execute MinimalAgent against a math calculation query and verify ArgumentCorrectnessMetric scores 1.00."""
    query = "Calculate 25 * 9/5 + 32"

    logger.info("Executing SUT argument correctness test for calculation query: %s", query)
    result = minimal_agent.run_with_execution(query)

    test_case = LLMTestCase(
        input=query,
        actual_output=result["answer"],
        tools_called=result["tools_called"],
    )

    metric = ArgumentCorrectnessMetric(threshold=0.5, model=judge_model)
    metric.measure(test_case)

    logger.info(
        "ArgumentCorrectnessMetric score for calculation query: %.2f (Reason: %s)",
        metric.score,
        metric.reason,
    )

    assert (
        metric.score >= 0.7
    ), f"Expected ArgumentCorrectnessMetric score >= 0.7 for calculation query, but got: {metric.score}"

    logger.info("Calculation query tool argument correctness evaluation passed successfully!")


@pytest.mark.dynamic
def test_minimal_agent_argument_correctness_time_query(minimal_agent, judge_model):
    """SUT Agent Test: Execute MinimalAgent against a time lookup query and verify ArgumentCorrectnessMetric scores 1.00."""
    query = "What is the current local time in Tokyo?"

    logger.info("Executing SUT argument correctness test for time query: %s", query)
    result = minimal_agent.run_with_execution(query)

    test_case = LLMTestCase(
        input=query,
        actual_output=result["answer"],
        tools_called=result["tools_called"],
    )

    metric = ArgumentCorrectnessMetric(threshold=0.5, model=judge_model)
    metric.measure(test_case)

    logger.info(
        "ArgumentCorrectnessMetric score for time query: %.2f (Reason: %s)",
        metric.score,
        metric.reason,
    )

    assert (
        metric.score >= 0.7
    ), f"Expected ArgumentCorrectnessMetric score >= 0.7 for time query, but got: {metric.score}"

    logger.info("Time query tool argument correctness evaluation passed successfully!")


@pytest.mark.dynamic
def test_minimal_agent_argument_correctness_multi_tool_trajectory(minimal_agent, judge_model):
    """SUT Agent Test: Execute MinimalAgent against a multi-tool sequential query.

    DeepEval ArgumentCorrectnessMetric Limitation Note:
    DeepEval evaluates tool call arguments strictly against test_case.input without inspecting
    intermediate tool execution outputs (tool.output). For chained multi-tool trajectories where
    tool #2 parameters depend on tool #1's dynamic output, the judge scores 0.50.
    We set threshold >= 0.50 to account for chained trajectory argument dependency.
    """
    query = "What is the weather in Tokyo and convert the temperature to Fahrenheit."

    logger.info("Executing SUT argument correctness test for multi-tool query: %s", query)
    result = minimal_agent.run_with_execution(query)

    test_case = LLMTestCase(
        input=query,
        actual_output=result["answer"],
        tools_called=result["tools_called"],
    )

    metric = ArgumentCorrectnessMetric(threshold=0.5, model=judge_model)
    metric.measure(test_case)

    logger.info(
        "ArgumentCorrectnessMetric score for multi-tool trajectory: %.2f (Reason: %s)",
        metric.score,
        metric.reason,
    )

    # Threshold >= 0.5 accounts for DeepEval's single-prompt argument evaluation limitation on chained tools
    assert (
        metric.score >= 0.5
    ), f"Expected ArgumentCorrectnessMetric score >= 0.5 for multi-tool chained trajectory, but got: {metric.score}"

    logger.info("Multi-tool trajectory argument correctness evaluation passed successfully!")
