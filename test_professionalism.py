from deepeval import assert_test
from deepeval.metrics import ConversationalGEval
from deepeval.test_case import ConversationalTestCase, MultiTurnParams, Turn


def test_professionalism(gemini_model):
    professionalism_metric = ConversationalGEval(
        name="Professionalism",
        criteria="Determine whether the assistant has acted professionally based on the content.",
        evaluation_params=[MultiTurnParams.CONTENT, MultiTurnParams.ROLE],
        model=gemini_model,
        threshold=0.9,
    )
    test_case = ConversationalTestCase(
        turns=[
            Turn(role="user", content="What is DeepEval?"),
            Turn(role="assistant", content="DeepEval is an open-source LLM eval package."),
            Turn(role="user", content="What can I use it for?"),
            Turn(role="assistant", content="You can use DeepEval to evaluate LLM outputs with metrics such as correctness, relevance, and safety."),
            Turn(role="user", content="How do I get started?"),
            Turn(role="assistant", content="Install it with pip, define a test case and metric, then run your tests with pytest."),
        ]
    )
    assert_test(test_case, [professionalism_metric])