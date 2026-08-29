import logging

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

import pytest

logger = logging.getLogger(__name__)


@pytest.mark.static
def test_correctness(ollama_judge_model, test_data_loader):
    test_data = test_data_loader("correctness.json")
    correctness_metric = GEval(
        name=test_data["name"],
        criteria=test_data["criteria"],
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        model=ollama_judge_model,
        threshold=test_data["threshold"],
    )
    test_case = LLMTestCase(
        input=test_data["input"],
        actual_output=test_data["actual_output"],
        expected_output=test_data["expected_output"],
    )
    assert_test(test_case, [correctness_metric])
