import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_and_save():
    """Trains a logistic regression pipeline and saves it as pkl

    Uses the best hyperparameters found during assignment 2 grid search.
    Loads training data from assignment 2 data folder.

    Returns:
        Pipeline: fitted sklearn pipeline
    """
    train = pd.read_csv("../Assignment-02/data/train.csv")
    X = train["message"]
    y = (train["label"] == "spam").astype(int)

    # best hyperparams from assignment 2 grid search
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
            (
                "clf",
                LogisticRegression(
                    C=100,
                    penalty="l2",
                    solver="liblinear",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X, y)
    joblib.dump(model, "best_model.pkl")
    print(f"saved best_model.pkl, classes: {model.classes_}")
    return model


if __name__ == "__main__":
    train_and_save()
