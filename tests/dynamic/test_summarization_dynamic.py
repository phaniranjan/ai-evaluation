import logging

from deepeval import assert_test
from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase

import pytest

logger = logging.getLogger(__name__)


@pytest.mark.dynamic
def test_summarization_dynamic(judge_model, test_data_loader, response_generator):
    """
    Test summarization quality with dynamically generated LLM summaries.
    
    This test:
    1. Generates a summary for the source input text using Gemini
    2. Evaluates the summary for truthfulness and keypoint coverage using SummarizationMetric
    """
    test_data = test_data_loader("summarization_dynamic.json")
    
    logger.info("Starting summarization dynamic test")
    logger.info(f"Input Document:\n{test_data['input']}")
    
    # Generate the actual summary output using LLM
    logger.info("Generating LLM summary...")
    system_instruction = "Summarize the provided text accurately and concisely."
    actual_output = response_generator(test_data["input"], system_instruction=system_instruction)
    
    logger.info(f"Generated Summary:\n{actual_output}")
    
    summarization_metric = SummarizationMetric(
        threshold=test_data.get("threshold", 0.7),
        model=judge_model,
    )
    
    test_case = LLMTestCase(
        input=test_data["input"],
        actual_output=actual_output,
    )
    
    logger.info(f"Threshold for evaluation: {test_data['threshold']}")
    assert_test(test_case, [summarization_metric])
    logger.info("Summarization dynamic test passed!")
