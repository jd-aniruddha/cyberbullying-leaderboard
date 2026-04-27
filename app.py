from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import json
from sklearn.metrics import accuracy_score, f1_score

app = Flask(__name__)
CORS(app)

# 🔥 Model logic
def predict(glucose):
    return "HIGH" if glucose > 140 else "LOW"

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json

    age = data["age"]
    bmi = data["bmi"]
    glucose = data["glucose"]
    bp = data["blood_pressure"]

    # Prediction
    prediction = predict(glucose)

    # Load dataset
    test = pd.read_csv("data/test.csv")
    gt = pd.read_csv("data/ground_truth.csv")

    test["prediction"] = test["glucose"].apply(predict)

    df = gt.merge(test[["id", "prediction"]], on="id")

    accuracy = accuracy_score(df["label"], df["prediction"])
    f1 = f1_score(df["label"], df["prediction"], pos_label="HIGH")

    # Update leaderboard
    leaderboard = [{
        "model": "Baseline Model",
        "accuracy": round(float(accuracy), 4),
        "f1_score": round(float(f1), 4),
        "last_prediction": prediction
    }]

    with open("leaderboard.json", "w") as f:
        json.dump(leaderboard, f, indent=2)

    return jsonify({
        "prediction": prediction,
        "accuracy": accuracy,
        "f1_score": f1
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

@app.route("/")
def home():
    return "Backend is running ✅"
