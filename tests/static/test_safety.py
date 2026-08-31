import logging

from deepeval import assert_test
from deepeval.metrics import BiasMetric, ToxicityMetric
from deepeval.test_case import LLMTestCase

import pytest

logger = logging.getLogger(__name__)


@pytest.mark.static
def test_safety(judge_model, test_data_loader):
    test_data = test_data_loader("safety.json")
    bias_metric = BiasMetric(
        threshold=test_data.get("bias_threshold", 0.7),
        model=judge_model,
    )
    toxicity_metric = ToxicityMetric(
        threshold=test_data.get("toxicity_threshold", 0.7),
        model=judge_model,
    )
    test_case = LLMTestCase(
        input=test_data["input"],
        actual_output=test_data["actual_output"],
    )
    assert_test(test_case, [bias_metric, toxicity_metric])
