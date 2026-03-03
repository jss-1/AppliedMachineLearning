import subprocess
import time
import pytest
import joblib
import requests
from score import score


SCORE_CASES = [
    # obvious spam
    {
        "id": "spam_1",
        "text": "WINNER! You won a free prize worth 10000 dollars! Call now to claim",
        "expected_prediction": True,
    },
    {
        "id": "spam_2",
        "text": "Congratulations you have been selected for a cash prize! Text WIN to 80808",
        "expected_prediction": True,
    },
    {
        "id": "spam_3",
        "text": "URGENT! Your mobile number has won a 2000 pound prize. Call 09061790121",
        "expected_prediction": True,
    },
    {
        "id": "spam_4",
        "text": "Congratulations ur awarded vouchers and guaranteed free entry to weekly draw txt to claim",
        "expected_prediction": True,
    },
    # obvious ham
    {
        "id": "ham_1",
        "text": "hey are you coming to class tomorrow",
        "expected_prediction": False,
    },
    {
        "id": "ham_2",
        "text": "can you pick up milk on your way home",
        "expected_prediction": False,
    },
    {
        "id": "ham_3",
        "text": "the assignment is due next monday",
        "expected_prediction": False,
    },
    {
        "id": "ham_4",
        "text": "i will be late for dinner sorry",
        "expected_prediction": False,
    },
]


@pytest.fixture(scope="module")
def model():
    """Loads the trained pipeline once for all tests to avoid reloading.

    Returns:
        Pipeline: fitted sklearn pipeline
    """
    return joblib.load("best_model.pkl")


@pytest.mark.parametrize("case", SCORE_CASES, ids=[c["id"] for c in SCORE_CASES])
def test_score_smoke(model, case):
    """Checks that score function runs without crashing

    Verifies the function returns a non-None result for each test case.

    Args:
        model (Pipeline): trained sklearn pipeline
        case (dict): test case with text, threshold, expected_prediction
    """
    result = score(case["text"], model, 0.5)
    assert result is not None, f"returned None for {case['id']}"


@pytest.mark.parametrize("case", SCORE_CASES, ids=[c["id"] for c in SCORE_CASES])
def test_score_prediction(model, case):
    """Checks that prediction matches expected value

    Compares the model prediction against the expected label
    for obvious spam and ham texts at default threshold.

    Args:
        model (Pipeline): trained sklearn pipeline
        case (dict): test case with text, threshold, expected_prediction
    """
    pred, prop = score(case["text"], model, 0.5)
    assert pred == case["expected_prediction"], \
        f"{case['id']}: expected {case['expected_prediction']}, got {pred} (propensity={prop:.4f})"


@pytest.mark.parametrize("case", SCORE_CASES, ids=[c["id"] for c in SCORE_CASES])
def test_score_output_types(model, case):
    """checks that output types are correct

    Prediction must be bool, propensity must be float.

    Args:
        model (Pipeline): trained sklearn pipeline
        case (dict): test case with text, threshold, expected_prediction
    """
    pred, prop = score(case["text"], model, 0.5)
    assert isinstance(pred, bool), \
        f"{case['id']}: prediction is {type(pred)}, expected bool"
    assert isinstance(prop, float), \
        f"{case['id']}: propensity is {type(prop)}, expected float"


@pytest.mark.parametrize("case", SCORE_CASES, ids=[c["id"] for c in SCORE_CASES])
def test_score_propensity_range(model, case):
    """checks that propensity is between 0 and 1

    Logistic regression probabilities should always be in [0, 1].

    Args:
        model (Pipeline): trained sklearn pipeline
        case (dict): test case with text, threshold, expected_prediction
    """
    _, prop = score(case["text"], model, 0.5)
    assert 0 <= prop <= 1, f"{case['id']}: propensity {prop:.4f} out of [0, 1]"


@pytest.mark.parametrize("case", SCORE_CASES, ids=[c["id"] for c in SCORE_CASES])
def test_score_threshold_edge(model, case):
    """Checks prediction at threshold boundaries 0 and 1

    threshold=0 should predict everything as spam since propensity >= 0.
    threshold=1 should predict everything as ham since logistic regression
    propensity is always strictly less than 1.

    Args:
        model (Pipeline): trained sklearn pipeline
        case (dict): test case from SCORE_CASES, only text field is used
    """
    pred_zero, prop = score(case["text"], model, 0.0)
    assert pred_zero == True, \
        f"{case['id']}: threshold=0 should be True, got {pred_zero} (propensity={prop:.4f})"

    pred_one, _ = score(case["text"], model, 1.0)
    assert pred_one == False, \
        f"{case['id']}: threshold=1 should be False, got {pred_one} (propensity={prop:.4f})"
