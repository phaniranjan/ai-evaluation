"""
LLM response generator for ai_evaluation tests.

Generates dynamic LLM responses that can be evaluated against expected outputs.
Uses Gemini API to generate responses to questions.
"""

from google import genai
from google.genai import types


def generate_response(question: str, api_key: str, model: str = "gemini-3.5-flash-lite") -> str:
    """
    Generate an LLM response to a question.
    
    Args:
        question: The question to answer
        api_key: Gemini API key
        model: Model to use (default: gemini-3.5-flash-lite)
    
    Returns:
        Generated response text
    """
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model,
        contents=question,
        config=types.GenerateContentConfig(
            max_output_tokens=400,
        ),
    )
    return resp.text.strip()


def generate_conversation_turn(
    user_message: str,
    api_key: str,
    model: str = "gemini-3.5-flash-lite",
    conversation_history: list = None,
) -> str:
    """
    Generate an assistant response in a multi-turn conversation.
    
    Args:
        user_message: The user's message
        api_key: Gemini API key
        model: Model to use (default: gemini-3.5-flash-lite)
        conversation_history: Previous turns in the conversation (list of {"role": "user"/"assistant", "content": "..."})
    
    Returns:
        Generated assistant response
    """
    client = genai.Client(api_key=api_key)
    
    # Build message history
    messages = []
    if conversation_history:
        for turn in conversation_history:
            messages.append(
                types.Content(
                    role="user" if turn["role"] == "user" else "model",
                    parts=[types.Part(text=turn["content"])],
                )
            )
    
    # Add current user message
    messages.append(
        types.Content(
            role="user",
            parts=[types.Part(text=user_message)],
        )
    )
    
    resp = client.models.generate_content(
        model=model,
        contents=messages,
        config=types.GenerateContentConfig(
            max_output_tokens=300,
        ),
    )
    return resp.text.strip()
