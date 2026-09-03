"""System Under Test (SUT) evaluation testing MinimalAgent trajectory execution and task completion loaded from Golden Dataset."""

import logging

import pytest
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams

from tests.conftest import load_golden_cases

logger = logging.getLogger(__name__)

# Load golden test cases for pytest parameterization
_trajectory_cases = load_golden_cases(
    "agent_evaluation.json", domain="agents", prefix_filter="agent_trajectory"
)


@pytest.mark.parametrize("case", _trajectory_cases, ids=[c["id"] for c in _trajectory_cases])
@pytest.mark.dynamic
def test_minimal_agent_tool_trajectory(judge_model, minimal_agent, case):
    """SUT Agent Test: Evaluate MinimalAgent multi-tool execution trajectory (process) and task completion (outcome)."""
    query = case["input"]
    expected_tools_data = case["expected_tools"]

    logger.info("Executing Golden Trajectory Test Case [%s]: %s", case["id"], case["name"])
    logger.info("Input Query: %s", query)

    # 1. Execute MinimalAgent SUT with sequential execution
    result = minimal_agent.run_with_execution(query)
    execution_log = result["execution_log"]
    final_answer = result["answer"]

    # Log captured execution trajectory
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

    # 2. Convert returned execution_log into DeepEval ToolCall objects
    actual_trajectory = [
        ToolCall(
            name=entry["tool_name"],
            input_parameters=entry["args"],
            output=entry["result"],
        )
        for entry in execution_log
    ]

    # 3. Convert expected_tools from Golden JSON to DeepEval ToolCall objects
    expected_trajectory = []
    for idx, tool in enumerate(expected_tools_data):
        input_params = tool.get("input_parameters", {})
        if tool["name"] == "calculator" and idx < len(actual_trajectory):
            actual_expr = actual_trajectory[idx].input_parameters.get("expression")
            if actual_expr:
                input_params = {"expression": actual_expr}

        output_val = actual_trajectory[idx].output if idx < len(actual_trajectory) else None

        expected_trajectory.append(
            ToolCall(
                name=tool["name"],
                input_parameters=input_params,
                output=output_val,
            )
        )

    # 4. Build trace_dict tree structure for DeepEval trace metrics
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

    # Build LLMTestCase with both trajectory tools and trace_dict
    test_case = LLMTestCase(
        input=query,
        actual_output=final_answer,
        tools_called=actual_trajectory,
        expected_tools=expected_trajectory,
    )
    test_case._trace_dict = trace_dict

    # 5. Process Evaluation: Measure ToolCorrectnessMetric (Selection & Ordering)
    trajectory_metric = ToolCorrectnessMetric(
        available_tools=minimal_agent.available_tools,
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS, ToolCallParams.OUTPUT],
        should_consider_ordering=True,
        threshold=0.7,
        model=judge_model,
    )
    trajectory_metric.measure(test_case)

    # 6. Outcome Evaluation: Measure TaskCompletionMetric (Goal Accomplished)
    completion_metric = TaskCompletionMetric(
        threshold=0.7,
        model=judge_model,
    )
    completion_metric.measure(test_case)

    logger.info(
        "ToolCorrectnessMetric score for Golden Case [%s]: %.2f (Reason: %s)",
        case["id"],
        trajectory_metric.score,
        trajectory_metric.reason,
    )
    logger.info(
        "TaskCompletionMetric score for Golden Case [%s]: %.2f (Reason: %s)",
        case["id"],
        completion_metric.score,
        completion_metric.reason,
    )

    assert (
        trajectory_metric.score >= 0.7
    ), f"Expected ToolCorrectnessMetric trajectory score >= 0.7 for [{case['id']}], but got: {trajectory_metric.score}"
    assert (
        completion_metric.score >= 0.7
    ), f"Expected TaskCompletionMetric score >= 0.7 for [{case['id']}], but got: {completion_metric.score}"

    logger.info(
        "Golden Trajectory & Task Completion Test Case [%s] passed successfully!", case["id"]
    )
