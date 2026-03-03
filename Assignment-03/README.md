# Assignment 3: Testing & Model Serving

**Course**: Applied Machine Learning
**Institute**: Chennai Mathematical Institute
**Student**: Jugaad Singh Sohal (MDS202421)

## Problem Statement

Build a scoring function for the SMS spam classifier from Assignment 2, serve it as a Flask endpoint, and write comprehensive unit and integration tests with coverage reporting.

## Score Function (`score.py`)

Takes a text string, a trained sklearn pipeline, and a classification threshold. Returns a boolean prediction and a float propensity score.

```python
def score(text: str, model: Pipeline, threshold: float) -> tuple[bool, float]
```

- Propensity is the probability of spam from `predict_proba`
- Prediction is `True` (spam) if propensity >= threshold

## Flask App (`app.py`)

POST endpoint at `/score` that accepts JSON with `text` (required) and `threshold` (optional, default 0.5). Returns JSON with `prediction` (bool) and `propensity` (float).

- 415 if request body is not JSON
- 422 if `text` is missing, not a string, or `threshold` is not a number
- Trains the model on startup if `best_model.pkl` is not found

## Testing (`test.py`)

### Unit Tests (parametrized over 8 test cases)

Test cases cover 4 obvious spam and 4 obvious ham texts. Each test function is parametrized over all cases:

| Test | What it checks |
|------|---------------|
| `test_score_smoke` | Function runs without crashing |
| `test_score_prediction` | Prediction matches expected label |
| `test_score_output_types` | Prediction is bool, propensity is float |
| `test_score_propensity_range` | Propensity is between 0 and 1 |
| `test_score_threshold_edge` | threshold=0 always predicts spam, threshold=1 always predicts ham |

### Integration Tests (Flask endpoint)

Flask app is launched as a subprocess on port 5001 for testing:

| Test | What it checks |
|------|---------------|
| `test_flask_score` | Endpoint response matches direct `score()` output |
| `test_flask_response_types` | Response JSON has correct types |
| `test_flask_errors` | Error status codes for bad requests (422, 415) |

### Coverage

```
Name       Stmts   Miss  Cover
------------------------------
score.py       6      0   100%
------------------------------
TOTAL          6      0   100%

60 passed
```

## Model

Logistic Regression pipeline (TF-IDF + classifier) using the best hyperparameters from Assignment 2:
- TF-IDF: max_features=5000, stop_words=english
- LogisticRegression: C=100, penalty=l2, solver=liblinear

Trained on the SMS Spam Collection dataset (Assignment 2 train split).

## Files

```
Assignment-03/
├── README.md
├── score.py              # Scoring function
├── train_model.py        # Trains and saves the pipeline
├── best_model.pkl        # Saved trained pipeline
├── app.py                # Flask app with /score endpoint
├── test.py               # Unit and integration tests
├── coverage.txt          # pytest coverage report
├── unit_test.log         # Unit test run log
└── full_test.log         # Full test run log
```

## How to Run

1. Install dependencies: `pip install flask pytest pytest-cov requests scikit-learn joblib`
2. Train the model: `python train_model.py`
3. Run the Flask app: `python app.py`
4. Run tests: `python -m pytest test.py -v`
5. Run with coverage: `python -m pytest test.py -v --cov=score --cov-report=term`
