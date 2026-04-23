import pandas as pd
import json
import os

gt = pd.read_csv("data/ground_truth.csv")

submission_files = os.listdir("submissions")
latest = sorted(submission_files)[-1]

sub = pd.read_csv(f"submissions/{latest}")

df = gt.merge(sub, on="id")

accuracy = (df["label"] == df["prediction"]).mean()
print("Accuracy:", accuracy)

leaderboard_file = "leaderboard.json"

if os.path.exists(leaderboard_file):
    with open(leaderboard_file, "r") as f:
        leaderboard = json.load(f)
else:
    leaderboard = []

leaderboard.append({
    "submission": latest,
    "accuracy": round(float(accuracy), 4)
})

leaderboard = sorted(leaderboard, key=lambda x: x["accuracy"], reverse=True)

with open(leaderboard_file, "w") as f:
    json.dump(leaderboard, f, indent=2)
from sklearn.metrics import f1_score

f1 = f1_score(df["label"], df["prediction"], pos_label="HIGH")
print("F1 Score:", f1)  
  
  
