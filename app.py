from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load saved model
model = joblib.load('gradient_boosting_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = data.get('features')

    if not features or len(features) != model.n_features_in_:
        return jsonify({'error': f'Invalid input length. Expected {model.n_features_in_} features.'}), 400

    prediction = model.predict([features])
    return jsonify({'predicted_strength': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
