import logging

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

import pytest

logger = logging.getLogger(__name__)


@pytest.mark.dynamic
def test_correctness_dynamic(judge_model, test_data_loader, response_generator):
    """
    Test correctness with dynamically generated LLM response.
    
    This test:
    1. Generates an LLM response to the input question
    2. Compares it against an expected output using GEval (LLM-as-judge)
    """
    test_data = test_data_loader("correctness_dynamic.json")
    
    logger.info("Starting correctness dynamic test")
    logger.info(f"Input: {test_data['input']}")
    
    # Generate the actual output using LLM
    logger.info("Generating LLM response...")
    actual_output = response_generator(test_data["input"])
    
    logger.info(f"Generated Output:\n{actual_output}")
    logger.info(f"Expected Output:\n{test_data['expected_output']}")
    
    correctness_metric = GEval(
        name=test_data["name"],
        criteria=test_data["criteria"],
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        model=judge_model,
        threshold=test_data["threshold"],
    )
    
    test_case = LLMTestCase(
        input=test_data["input"],
        actual_output=actual_output,
        expected_output=test_data["expected_output"],
    )
    
    logger.info(f"Threshold for evaluation: {test_data['threshold']}")
    assert_test(test_case, [correctness_metric])
    logger.info("Correctness dynamic test passed!")
