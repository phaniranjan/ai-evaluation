"""System Under Test (SUT) evaluation testing MinimalAgent infinite tool loop detection loaded from Golden Dataset."""

import logging

import pytest
from deepeval.metrics import AgentLoopDetectionMetric
from deepeval.test_case import LLMTestCase

from tests.conftest import load_golden_cases

logger = logging.getLogger(__name__)

# Load golden loop test cases for pytest parameterization
_loop_cases = load_golden_cases(
    "agent_trajectory.json", domain="agents", prefix_filter="agent_loop"
)


@pytest.mark.parametrize("case", _loop_cases, ids=[c["id"] for c in _loop_cases])
@pytest.mark.dynamic
def test_minimal_agent_loop_detection(minimal_agent, case):
    """SUT Agent Test: Execute MinimalAgent against Golden Dataset loop scenario and verify AgentLoopDetectionMetric flags defect."""
    query = case["input"]

    logger.info("Executing Golden Loop Detection Test Case [%s]: %s", case["id"], case["name"])
    logger.info("Input Query: %s", query)

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
        final_answer if final_answer else "Execution terminated after maximum tool retry attempts."
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
        "AgentLoopDetectionMetric score for Golden Case [%s]: %.2f (Reason: %s)",
        case["id"],
        metric.score,
        metric.reason,
    )

    # Assert that the real SUT trajectory contains a loop defect (>= 3 identical calls) and metric flags it (< 0.5)
    assert (
        len(execution_log) >= 3
    ), f"Expected SUT to execute >= 3 tool calls for [{case['id']}], but got: {len(execution_log)}"
    assert (
        metric.score < 0.5
    ), f"Expected AgentLoopDetectionMetric score < 0.5 for [{case['id']}], but got: {metric.score}"

    logger.info("Golden Loop Detection Test Case [%s] passed successfully!", case["id"])
