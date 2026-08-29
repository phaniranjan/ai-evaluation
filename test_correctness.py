from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams


def test_correctness(gemini_model, test_data_loader):
    test_data = test_data_loader("correctness.json")
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        model=gemini_model,
        threshold=0.9,
    )
    test_case = LLMTestCase(
        input=test_data["input"],
        actual_output=test_data["actual_output"],
        expected_output=test_data["expected_output"],
    )
    assert_test(test_case, [correctness_metric])