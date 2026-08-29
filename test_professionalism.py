from deepeval import assert_test
from deepeval.metrics import ConversationalGEval
from deepeval.test_case import ConversationalTestCase, MultiTurnParams, Turn


def test_professionalism(gemini_model, test_data_loader):
    test_data = test_data_loader("professionalism.json")
    professionalism_metric = ConversationalGEval(
        name=test_data["name"],
        criteria=test_data["criteria"],
        evaluation_params=[MultiTurnParams.CONTENT, MultiTurnParams.ROLE],
        model=gemini_model,
        threshold=test_data["threshold"],
    )
    test_case = ConversationalTestCase(
        turns=[Turn(**turn) for turn in test_data["turns"]]
    )
    assert_test(test_case, [professionalism_metric])