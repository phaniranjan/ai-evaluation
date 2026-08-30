import logging

from deepeval import assert_test
from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase

import pytest

logger = logging.getLogger(__name__)


@pytest.mark.static
def test_summarization(judge_model, test_data_loader):
    test_data = test_data_loader("summarization.json")
    summarization_metric = SummarizationMetric(
        threshold=test_data.get("threshold", 0.7),
        model=judge_model,
    )
    test_case = LLMTestCase(
        input=test_data["input"],
        actual_output=test_data["actual_output"],
    )
    assert_test(test_case, [summarization_metric])
