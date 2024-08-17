from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase


def test_case():
    arm = AnswerRelevancyMetric(threshold=0.7)
    cr = ContextualRelevancyMetric(threshold=0.7)
    cp = ContextualPrecisionMetric(threshold=0.7)
    hm = HallucinationMetric(threshold=0.7)

    test_case = LLMTestCase(
        input="What if these shoes doesn't fit me?",
        actual_output="We offer a 30-day full refund at some extra costs.",
        expected_output="We offer a 30-day full refund at no extra costs.",
        retrieval_context=[
            "All the customers are eligilble fro a 30-day full refund at no costs."
        ],
        context=[
            "All the customers are eligilble fro a 30-day full refund at no costs."
        ],
    )

    assert_test(test_case, [arm, cr, cp, hm])
