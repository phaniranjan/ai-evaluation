from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams


def test_correctness_dynamic(gemini_model, test_data_loader, response_generator):
    """
    Test correctness with dynamically generated LLM response.
    
    This test:
    1. Generates an LLM response to the input question
    2. Compares it against an expected output using GEval (LLM-as-judge)
    """
    test_data = test_data_loader("correctness_dynamic.json")
    
    # Generate the actual output using LLM
    actual_output = response_generator(test_data["input"])
    
    print(f"\n=== Correctness (Dynamic) Test ===")
    print(f"Input: {test_data['input']}")
    print(f"\nGenerated Output:\n{actual_output}")
    print(f"\nExpected Output:\n{test_data['expected_output']}")
    
    correctness_metric = GEval(
        name=test_data["name"],
        criteria=test_data["criteria"],
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        model=gemini_model,
        threshold=test_data["threshold"],
    )
    
    test_case = LLMTestCase(
        input=test_data["input"],
        actual_output=actual_output,
        expected_output=test_data["expected_output"],
    )
    
    assert_test(test_case, [correctness_metric])
