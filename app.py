from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

app = Flask(__name__)
CORS(app)

test = pd.read_csv("data/test.csv")
gt = pd.read_csv("data/ground_truth.csv")

X = test[["age", "bmi", "glucose", "blood_pressure"]].values
y = gt["label"].map({"LOW": 0, "HIGH": 1}).values

model = LogisticRegression(solver="liblinear")
model.fit(X, y)

preds = model.predict(X)

labels = np.where(preds == 1, "HIGH", "LOW")

df = gt.copy()
df["prediction"] = labels

ACCURACY = float(accuracy_score(df["label"], df["prediction"]))
F1 = float(f1_score(df["label"], df["prediction"], pos_label="HIGH"))

@app.route("/")
def home():
    return "Backend is running"

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json

    user_input = np.array([[ 
        data["age"],
        data["bmi"],
        data["glucose"],
        data["blood_pressure"]
    ]])

    pred = model.predict(user_input)[0]

    return jsonify({
        "prediction": "HIGH" if pred == 1 else "LOW",
        "accuracy": round(ACCURACY, 4),
        "f1_score": round(F1, 4)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, threaded=True)
