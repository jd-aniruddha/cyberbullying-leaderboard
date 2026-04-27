from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

app = Flask(__name__)
CORS(app)

# 🔹 Home route (for testing)
@app.route("/")
def home():
    return "Backend is running ✅"


# 🔹 Prediction API
@app.route("/predict", methods=["POST"])
def predict_api():

    # Get user input
    data = request.json

    age = data["age"]
    bmi = data["bmi"]
    glucose = data["glucose"]
    bp = data["blood_pressure"]

    # 🔹 Load dataset
    test = pd.read_csv("data/test.csv")
    gt = pd.read_csv("data/ground_truth.csv")

    # 🔹 Prepare training data
    X = test[["age", "bmi", "glucose", "blood_pressure"]]
    y = gt["label"].map({"LOW": 0, "HIGH": 1})   # encode labels

    # 🔹 Train model
    model = LogisticRegression()
    model.fit(X, y)

    # 🔹 Predict on dataset
    test["prediction"] = model.predict(X)
    test["prediction"] = test["prediction"].map({0: "LOW", 1: "HIGH"})

    # 🔹 Evaluate model
    df = gt.merge(test[["id", "prediction"]], on="id")

    accuracy = accuracy_score(df["label"], df["prediction"])
    f1 = f1_score(df["label"], df["prediction"], pos_label="HIGH")

    # 🔹 Predict for user input
    user_df = pd.DataFrame(
        [[age, bmi, glucose, bp]],
        columns=["age", "bmi", "glucose", "blood_pressure"]
    )

    user_pred = model.predict(user_df)[0]
    user_pred = "HIGH" if user_pred == 1 else "LOW"

    # 🔹 Return result
    return jsonify({
        "prediction": user_pred,
        "accuracy": round(float(accuracy), 4),
        "f1_score": round(float(f1), 4)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
