import logging

from deepeval import assert_test
from deepeval.metrics import ConversationalGEval
from deepeval.test_case import ConversationalTestCase, MultiTurnParams, Turn

import pytest

logger = logging.getLogger(__name__)


@pytest.mark.static
def test_multi_turn(judge_model, test_data_loader):
    test_data = test_data_loader("multi_turn.json")
    professionalism_metric = ConversationalGEval(
        name=test_data["name"],
        criteria=test_data["criteria"],
        evaluation_params=[MultiTurnParams.CONTENT, MultiTurnParams.ROLE],
        model=judge_model,
        threshold=test_data["threshold"],
    )
    test_case = ConversationalTestCase(
        turns=[Turn(**turn) for turn in test_data["turns"]]
    )
    assert_test(test_case, [professionalism_metric])
