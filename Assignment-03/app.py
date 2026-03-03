import os
import logging
import argparse
from flask import Flask, request, jsonify
import joblib
from score import score

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

if not os.path.exists('best_model.pkl'):
    logger.info('best_model.pkl not found, training model')
    from train_model import train_and_save
    train_and_save()

model = joblib.load('best_model.pkl')
logger.info('model loaded')


@app.route('/score', methods=['POST'])
def score_text():
    """scores input text for spam classification

    Args:
        request body (json): must contain 'text' (str), optional 'threshold' (float, default 0.5)

    Returns:
        json: {'prediction': bool, 'propensity': float}
    """
    data = request.get_json()

    if data is None:
        logger.warning('request body is not json')
        return jsonify({'error': 'request must be json'}), 415

    if 'text' not in data:
        logger.warning('missing text field')
        return jsonify({'error': 'text field required'}), 422

    text = data['text']
    if not isinstance(text, str):
        logger.warning(f'text field is not string, got {type(text).__name__}')
        return jsonify({'error': 'text must be a string'}), 422

    threshold = data.get('threshold', 0.5)
    if not isinstance(threshold, (int, float)):
        logger.warning(f'bad threshold type: {type(threshold).__name__}')
        return jsonify({'error': 'threshold must be a number'}), 422

    logger.info(f'scoring text ({len(text)} chars), threshold={threshold}')
    prediction, propensity = score(text, model, threshold)
    logger.info(f'result: prediction={prediction}, propensity={propensity:.4f}')

    return jsonify({'prediction': prediction, 'propensity': propensity})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    app.run(port=args.port)
