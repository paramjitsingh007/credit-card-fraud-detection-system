from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("credit_card_model.joblib")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert input to numpy array
        data = [float(x) for x in request.form.values()]
        prediction = model.predict([data])[0]

        result = "✅ Normal Transaction" if prediction == 0 else "⚠️ Fraudulent Transaction"

        # Return JSON response instead of rendering HTML
        return jsonify({'prediction': result})
    except Exception as e:
        print("Error:", e)
        return jsonify({'prediction': f"Error: {e}"})

if __name__ == "__main__":
    app.run(debug=True)
