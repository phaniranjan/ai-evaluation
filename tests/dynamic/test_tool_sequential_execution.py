"""System Under Test (SUT) evaluation testing MinimalAgent multi-tool sequential execution."""

import logging

import pytest

logger = logging.getLogger(__name__)


@pytest.mark.dynamic
def test_minimal_agent_sequential_tool_execution(minimal_agent):
    """SUT Agent Test: Verify MinimalAgent executes get_weather first, then passes the temperature to calculator sequentially."""
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

    # 1. Verify that both tools were actually executed
    assert len(execution_log) == 2, f"Expected 2 tools to be executed, but got {len(execution_log)}"

    # 2. Verify that get_weather is called first
    step1 = execution_log[0]
    assert (
        step1["tool_name"] == "get_weather"
    ), f"Expected first tool call to be 'get_weather', but got '{step1['tool_name']}'"
    assert (
        "tokyo" in str(step1["args"].get("location", "")).lower()
    ), f"Expected get_weather location to be Tokyo, but got {step1['args']}"

    # 3. Verify that calculator is called second
    step2 = execution_log[1]
    assert (
        step2["tool_name"] == "calculator"
    ), f"Expected second tool call to be 'calculator', but got '{step2['tool_name']}'"

    # 4. Verify calculator receives the appropriate value (25°C) derived from weather tool result
    calc_expression = str(step2["args"].get("expression", ""))
    assert (
        "25" in calc_expression
    ), f"Expected calculator expression to contain temperature '25' derived from weather result, but got '{calc_expression}'"

    # 5. Verify final response is produced
    assert final_answer, "Expected agent to produce a non-empty final answer"
    logger.info("Sequential tool execution test verified successfully!")

