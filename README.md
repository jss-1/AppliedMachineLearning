# Applied Machine Learning

**Course**: Applied Machine Learning
**Institute**: Chennai Mathematical Institute
**Semester**: January 2026 - May 2026

## Assignments

### 1. [SMS Spam Filter](./Assignment-01/)

Build a spam filter for SMS messages that classifies texts as "ham" (legitimate) or "spam". The dataset has class imbalance (~13% spam) and asymmetric costs where false positives (blocking legitimate messages) are 10x more costly than false negatives. Trained and compared Logistic Regression, Naive Bayes, and SVM models, selecting Naive Bayes as the best performer based on F1, AUPRC, and weighted cost.

### 2. [Experiment Tracking](./Assignment-02/)

Extend the SMS spam classification workflow with DVC for data version control and MLflow for experiment tracking. Track two data split versions (different random seeds) with DVC, train Logistic Regression, Naive Bayes, and Linear SVM on both versions, and log all hyperparameters, metrics (AUCPR), and models to MLflow with a SQLite backend. Checkout registered models from the MLflow registry and verify via REST API.

### 3. [Testing & Model Serving](./Assignment-03/)

Build a scoring function for the spam classifier, serve it as a Flask POST endpoint (`/score`), and write comprehensive tests. Unit tests cover smoke, format, propensity range, prediction accuracy, and threshold edge cases using parametrized test cases. Integration tests launch the Flask app and verify endpoint responses match direct function output. Includes error handling tests for bad requests (415, 422) and a coverage report.
