import logging

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

import pytest

logger = logging.getLogger(__name__)


@pytest.mark.dynamic
def test_single_turn_dynamic(judge_model, test_data_loader, response_generator):
    """
    Test correctness and relevance for dynamically generated single-turn Q&A.
    
    This test:
    1. Generates an LLM response to the input question
    2. Evaluates correctness against expected output (GEval)
    3. Evaluates answer relevancy against input prompt (AnswerRelevancyMetric)
    """
    test_data = test_data_loader("single_turn.json")
    
    logger.info("Starting single-turn dynamic test")
    logger.info(f"Input: {test_data['input']}")
    
    # Generate the actual output using LLM
    logger.info("Generating LLM response...")
    system_instruction = test_data.get(
        "system_instruction",
        "Provide a concise, direct, and factual answer in a clear paragraph without conversational filler, disclaimers, or follow-up questions.",
    )
    actual_output = response_generator(test_data["input"], system_instruction=system_instruction)
    
    logger.info(f"Generated Output:\n{actual_output}")
    logger.info(f"Expected Output:\n{test_data['expected_output']}")
    
    correctness_metric = GEval(
        name=test_data["name"],
        criteria=test_data["criteria"],
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        model=judge_model,
        threshold=test_data["threshold"],
    )

    relevancy_metric = AnswerRelevancyMetric(
        threshold=test_data.get("relevancy_threshold", 0.7),
        model=judge_model,
    )
    
    test_case = LLMTestCase(
        input=test_data["input"],
        actual_output=actual_output,
        expected_output=test_data["expected_output"],
    )
    
    logger.info(f"Threshold for evaluation: {test_data['threshold']}")
    assert_test(test_case, [correctness_metric, relevancy_metric])
    logger.info("Single-turn dynamic test passed!")
