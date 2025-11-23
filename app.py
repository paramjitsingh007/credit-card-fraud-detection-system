from flask import Flask, render_template, request
import pandas as pd
import joblib
from geopy.distance import geodesic

app = Flask(__name__)

# Load model & encoders
model = joblib.load("fraud_detection_model.jb")
encoder = joblib.load("label_encoder.jb")

def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None

    if request.method == "POST":

        merchant = request.form.get("merchant", "").strip()
        category = request.form.get("category", "").strip()
        amt = request.form.get("amt", "0")
        lat = float(request.form.get("lat", 0))
        long = float(request.form.get("long", 0))
        merch_lat = float(request.form.get("merch_lat", 0))
        merch_long = float(request.form.get("merch_long", 0))
        hour = int(request.form.get("hour", 12))
        day = int(request.form.get("day", 15))
        month = int(request.form.get("month", 6))
        gender = request.form.get("gender", "")
        cc_num = request.form.get("cc_num", "").strip()

        # Required fields check
        if not merchant or not category or not cc_num:
            prediction_result = "Please fill all required fields."
            return render_template("index.html", result=prediction_result)

        # Compute distance
        distance = calculate_distance(lat, long, merch_lat, merch_long)

        # Prepare input dataframe
        input_data = pd.DataFrame([[merchant, category, float(amt), distance,
                                    hour, day, month, gender, cc_num]],
                                  columns=['merchant', 'category', 'amt', 'distance',
                                           'hour', 'day', 'month', 'gender', 'cc_num'])

        # Encode categorical columns
        categorical_cols = ['merchant', 'category', 'gender']
        for col in categorical_cols:
            try:
                input_data[col] = encoder[col].transform(input_data[col])
            except:
                input_data[col] = -1

        # Hash CC number for privacy
        input_data['cc_num'] = input_data['cc_num'].apply(lambda x: hash(x) % (10 ** 6))

        # Prediction
        pred = model.predict(input_data)[0]
        prediction_result = "Fraudulent Transaction" if pred == 1 else "Legitimate Transaction"

    return render_template("index.html", result=prediction_result)


if __name__ == "__main__":
    app.run(debug=True)

