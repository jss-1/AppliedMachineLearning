import numpy as np
from sklearn.pipeline import Pipeline


def score(text: str, model: Pipeline, threshold: float) -> tuple[bool, float]:
    """scores a trained model on a given text

    Args:
        text (str): input text to classify
        model (sklearn.pipeline.Pipeline): trained sklearn pipeline with vectorizer and classifier
        threshold (float): classification threshold for spam prediction

    Returns:
        tuple[bool, float]: (prediction, propensity) where prediction is bool and propensity is float
    """
    propensity = model.predict_proba([text])[0][1]
    prediction = propensity >= threshold
    return bool(prediction), float(propensity)
