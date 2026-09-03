"""System Under Test (SUT) evaluation testing MinimalAgent infinite tool loop detection with DeepEval AgentLoopDetectionMetric."""

import logging

import pytest
from deepeval.metrics import AgentLoopDetectionMetric
from deepeval.test_case import LLMTestCase

logger = logging.getLogger(__name__)


@pytest.mark.dynamic
def test_minimal_agent_loop_detection(minimal_agent):
    """SUT Agent Test: Execute MinimalAgent against a retry-triggering prompt and verify AgentLoopDetectionMetric flags the loop defect."""
    query = "What is the weather in Retry_Tokyo? Keep retrying get_weather until it succeeds."

    logger.info("Executing SUT loop detection test for query: %s", query)
    result = minimal_agent.run_with_execution(query)

    execution_log = result["execution_log"]
    final_answer = result["answer"]

    logger.info("Captured Execution Trajectory (%d steps):", len(execution_log))
    for entry in execution_log:
        logger.info(
            "Step #%d | Tool: %s | Args: %s | Result: %s",
            entry["order"],
            entry["tool_name"],
            entry["args"],
            entry["result"],
        )

    # 1. Convert real SUT execution_log to DeepEval _trace_dict tree structure
    children = [
        {
            "name": entry["tool_name"],
            "type": "tool",
            "input": entry["args"],
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

    # 2. Build LLMTestCase with captured SUT trace_dict
    actual_output = (
        final_answer
        if final_answer
        else "Execution terminated after maximum tool retry attempts."
    )
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
    )
    test_case._trace_dict = trace_dict

    # 3. Instantiate built-in deterministic AgentLoopDetectionMetric
    metric = AgentLoopDetectionMetric(threshold=0.5, repetition_threshold=3, strict_mode=True)
    metric.measure(test_case)

    logger.info(
        "AgentLoopDetectionMetric score on real SUT trajectory: %.2f (Reason: %s)",
        metric.score,
        metric.reason,
    )

    # Assert that the real SUT trajectory contains a loop defect (>= 3 identical calls) and metric flags it (< 0.5)
    assert (
        len(execution_log) >= 3
    ), f"Expected SUT to execute >= 3 tool calls, but got: {len(execution_log)}"
    assert (
        metric.score < 0.5
    ), f"Expected AgentLoopDetectionMetric score < 0.5 for real SUT loop defect, but got: {metric.score}"

    logger.info("MinimalAgent loop detection evaluation passed successfully!")
