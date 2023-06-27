from flask import Flask, jsonify, request
from setfit import SetFitModel # Import Model

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()

    # To Handle Error like "text" data not exits in JSON object.
    if 'text' not in data:
        return jsonify({'error': 'Invalid request. Missing "text" parameter.'})

    text = data['text']
    model = SetFitModel.from_pretrained("StatsGary/setfit-ft-sentinent-eval")
    # Perform sentiment analysis on the text
    sentiment = model([text])

    #To handle Error
    # raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
    # TypeError: Object of type Tensor is not JSON serializable
    sentiment = sentiment.tolist()

    if sentiment[0] == 1:
        sentiment = 'positive'
    elif sentiment[0] == 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    respon = {
        'sentiment': sentiment
    }

    return jsonify(respon)


if __name__ == '__main__':
    app.run(debug=True)
