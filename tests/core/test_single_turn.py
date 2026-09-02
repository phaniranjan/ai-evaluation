"""System Under Test (SUT) evaluation testing single-turn Q&A using Golden Dataset cases."""

import logging

import pytest
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from tests.conftest import load_golden_cases

logger = logging.getLogger(__name__)

_single_turn_cases = load_golden_cases("single_turn.json", domain="core")


@pytest.mark.parametrize("case", _single_turn_cases, ids=[c["id"] for c in _single_turn_cases])
@pytest.mark.dynamic
def test_single_turn_dynamic(judge_model, response_generator, case):
    """SUT Core Test: Evaluate single-turn Q&A correctness and relevancy against Golden Dataset reference answers."""
    user_query = case["input"]
    reference_answer = case["reference_answer"]

    logger.info("Executing Golden Single-Turn Test Case [%s]: %s", case["id"], case["name"])
    logger.info("Input Query: %s", user_query)

    system_instruction = "Provide a concise, direct, and factual answer in a clear paragraph without conversational filler."
    actual_output = response_generator(user_query, system_instruction=system_instruction)

    logger.info("Generated Output:\n%s", actual_output)
    logger.info("Reference Answer:\n%s", reference_answer)

    correctness_metric = GEval(
        name="Single-Turn Correctness",
        criteria="Determine if the actual output is factually correct and consistent with the expected reference answer.",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        model=judge_model,
        threshold=0.7,
    )

    relevancy_metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=judge_model,
    )

    test_case = LLMTestCase(
        input=user_query,
        actual_output=actual_output,
        expected_output=reference_answer,
    )

    correctness_metric.measure(test_case)
    relevancy_metric.measure(test_case)

    logger.info(
        "GEval Correctness score for Golden Case [%s]: %.2f (Reason: %s)",
        case["id"],
        correctness_metric.score,
        correctness_metric.reason,
    )
    logger.info(
        "AnswerRelevancyMetric score for Golden Case [%s]: %.2f (Reason: %s)",
        case["id"],
        relevancy_metric.score,
        relevancy_metric.reason,
    )

    assert (
        correctness_metric.score >= 0.7
    ), f"Expected GEval score >= 0.7 for [{case['id']}], but got: {correctness_metric.score}"
    assert (
        relevancy_metric.score >= 0.7
    ), f"Expected AnswerRelevancyMetric >= 0.7 for [{case['id']}], but got: {relevancy_metric.score}"

    logger.info("Golden Single-Turn Test Case [%s] passed successfully!", case["id"])
