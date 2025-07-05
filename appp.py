from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("model1 (2).pkl")

@app.route("/")
def home():
    return render_template("indexx.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect inputs from form
        data = [
            float(request.form["pressure"]),
            float(request.form["temparature"]),
            float(request.form["dewpoint"]),
            float(request.form["humidity"]),
            float(request.form["cloud"]),
            float(request.form["sunshine"]),
            float(request.form["winddirection"]),
            float(request.form["windspeed"])
        ]
        # Predict
        prediction = model.predict([data])[0]
        result = "🌧️ Rainfall Expected" if prediction == 1 else "☀️ No Rainfall Expected"
        return render_template("index.html", prediction=result)
    except Exception as e:
        return jsonify({"error": str(e)})

# API route
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        features = [float(x) for x in data["features"]]
        prediction = model.predict([features])[0]
        result = "Rainfall Expected" if prediction == 1 else "No Rainfall"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)


# from flask import Flask, render_template, request
# import numpy as np
# import pandas as pd
# import joblib

# app = Flask(__name__)

# # Load model and scaler
# model = joblib.load("model1 (2).pkl")[1]  # RandomForest model at index 1
# loaded = joblib.load("model1 (2).pkl")
# # If the loaded object is a list, get the model from it
# if isinstance(loaded, list):
#     model = loaded[0]
# else:
#     model = loaded

# scaler = joblib.load("scaler.pkl")

# # Prediction route
# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = None
#     if request.method == "POST":
#         try:
#             # Get form data and convert to float
#             data = [float(request.form[field]) for field in [
#                 "season", "weather", "temp", "humidity", "windspeed", "casual",
#                 "day", "month", "year", "weekday", "am_or_pm", "holidays"
#             ]]
            
#             # Prepare DataFrame
#             columns = ['season', 'weather', 'temp', 'humidity', 'windspeed', 'casual',
#                        'day', 'month', 'year', 'weekday', 'am_or_pm', 'holidays']
#             input_df = pd.DataFrame([data], columns=columns)
            
#             # Scale input
#             scaled_input = scaler.transform(input_df)
            
#             # Predict
#             result = model.predict(scaled_input)
#             prediction = f"Predicted Ride Demand: {round(result[0], 2)}"
#         except Exception as e:
#             prediction = f"Error: {e}"
    
#     return render_template("index.html", prediction=prediction)

# if __name__ == "__main__":
#     app.run(debug=True)
# # 