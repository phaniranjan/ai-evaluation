from deepeval import assert_test
from deepeval.metrics import ConversationalGEval
from deepeval.test_case import ConversationalTestCase, MultiTurnParams, Turn


def test_professionalism_dynamic(gemini_model, test_data_loader, conversation_generator):
    """
    Test professionalism with dynamically generated LLM conversation.
    
    This test:
    1. Generates a multi-turn conversation with LLM-generated assistant responses
    2. Evaluates the conversation using ConversationalGEval (LLM-as-judge)
    """
    test_data = test_data_loader("professionalism_dynamic.json")
    
    # Generate the dynamic conversation
    turns_data = conversation_generator(
        initial_prompt="You are a helpful assistant.",
        turns=test_data["user_turns"]
    )
    
    print(f"\n=== Professionalism (Dynamic) Test ===")
    print(f"Generated Conversation:")
    for turn in turns_data:
        role = "User" if turn["role"] == "user" else "Assistant"
        print(f"\n{role}: {turn['content']}")
    
    professionalism_metric = ConversationalGEval(
        name=test_data["name"],
        criteria=test_data["criteria"],
        evaluation_params=[MultiTurnParams.CONTENT, MultiTurnParams.ROLE],
        model=gemini_model,
        threshold=test_data["threshold"],
    )
    
    test_case = ConversationalTestCase(
        turns=[Turn(**turn) for turn in turns_data]
    )
    
    assert_test(test_case, [professionalism_metric])
