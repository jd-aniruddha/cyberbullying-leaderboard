import pandas as pd
import json
import os
from sklearn.metrics import accuracy_score, f1_score

# Load data
gt = pd.read_csv("data/ground_truth.csv")

# Get latest submission
submission_files = os.listdir("submissions")
latest = sorted(submission_files)[-1]

sub = pd.read_csv(f"submissions/{latest}")

# Merge
df = gt.merge(sub, on="id")

# Metrics
accuracy = accuracy_score(df["label"], df["prediction"])
f1 = f1_score(df["label"], df["prediction"], pos_label="HIGH")

print("Accuracy:", accuracy)
print("F1 Score:", f1)

# Load leaderboard
leaderboard_file = "leaderboard.json"

if os.path.exists(leaderboard_file):
    with open(leaderboard_file, "r") as f:
        leaderboard = json.load(f)
else:
    leaderboard = []

# Add result
leaderboard.append({
    "submission": latest,
    "accuracy": round(float(accuracy), 4),
    "f1_score": round(float(f1), 4)
})

# Sort safely (prevents crash if key missing)
leaderboard = sorted(
    leaderboard,
    key=lambda x: x.get("f1_score", 0),
    reverse=True
)

# Save leaderboard ✅
with open(leaderboard_file, "w") as f:
    json.dump(leaderboard, f, indent=2)
