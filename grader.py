import pandas as pd
import json
import os
from sklearn.metrics import accuracy_score, f1_score

# Load data
test = pd.read_csv("data/test.csv")
gt = pd.read_csv("data/ground_truth.csv")

leaderboard = []

# 🔥 1. BASELINE MODEL
test["prediction"] = test["glucose"].apply(lambda x: "HIGH" if x > 140 else "LOW")

baseline_df = gt.merge(test[["id", "prediction"]], on="id")

baseline_acc = accuracy_score(baseline_df["label"], baseline_df["prediction"])
baseline_f1 = f1_score(baseline_df["label"], baseline_df["prediction"], pos_label="HIGH")

leaderboard.append({
    "submission": "BASELINE",
    "accuracy": round(float(baseline_acc), 4),
    "f1_score": round(float(baseline_f1), 4)
})

# 🔥 2. USER SUBMISSIONS
submission_files = os.listdir("submissions")

for file in submission_files:
    sub = pd.read_csv(f"submissions/{file}")

    df = gt.merge(sub, on="id")

    acc = accuracy_score(df["label"], df["prediction"])
    f1 = f1_score(df["label"], df["prediction"], pos_label="HIGH")

    leaderboard.append({
        "submission": file,
        "accuracy": round(float(acc), 4),
        "f1_score": round(float(f1), 4)
    })

# 🔥 Sort leaderboard
leaderboard = sorted(
    leaderboard,
    key=lambda x: x["f1_score"],
    reverse=True
)

# Save
with open("leaderboard.json", "w") as f:
    json.dump(leaderboard, f, indent=2)

print("Leaderboard updated successfully!")
