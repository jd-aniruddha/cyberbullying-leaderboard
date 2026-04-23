import pandas as pd
import json
from sklearn.metrics import accuracy_score, f1_score

# Load data
test = pd.read_csv("data/test.csv")
gt = pd.read_csv("data/ground_truth.csv")

# 🔥 Generate predictions (baseline model)
# Rule: glucose > 140 → HIGH
test["prediction"] = test["glucose"].apply(lambda x: "HIGH" if x > 140 else "LOW")

# Merge with ground truth
df = gt.merge(test[["id", "prediction"]], on="id")

# Metrics
accuracy = accuracy_score(df["label"], df["prediction"])
f1 = f1_score(df["label"], df["prediction"], pos_label="HIGH")

print("Accuracy:", accuracy)
print("F1 Score:", f1)

# Save result to leaderboard
leaderboard = [{
    "model": "Baseline Model",
    "accuracy": round(float(accuracy), 4),
    "f1_score": round(float(f1), 4)
}]

with open("leaderboard.json", "w") as f:
    json.dump(leaderboard, f, indent=2)
