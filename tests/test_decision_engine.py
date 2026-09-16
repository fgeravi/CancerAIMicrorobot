import pytest

from src.ai.classifier import ClassificationResult
from src.ai.decision_engine import DecisionEngine


def result(probability):
    return ClassificationResult(
        cancer_probability=probability,
        explanation={},
    )


def test_target_decision():

    engine = DecisionEngine(
        target_threshold=0.85,
        pass_threshold=0.30,
    )

    decision = engine.decide(result(0.90))

    assert decision.action == "TARGET"


def test_pass_decision():

    engine = DecisionEngine(
        target_threshold=0.85,
        pass_threshold=0.30,
    )

    decision = engine.decide(result(0.20))

    assert decision.action == "PASS"


def test_uncertain_decision():

    engine = DecisionEngine(
        target_threshold=0.85,
        pass_threshold=0.30,
    )

    decision = engine.decide(result(0.60))

    assert decision.action == "UNCERTAIN"


def test_invalid_thresholds():

    with pytest.raises(ValueError):
        DecisionEngine(
            target_threshold=0.20,
            pass_threshold=0.30,
        )
