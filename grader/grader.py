import pandas as pd
import os
import json
from datetime import datetime
from sklearn.metrics import f1_score

# 🔍 Find submission file
def find_submission():
    files = [f for f in os.listdir("submissions") if f.endswith(".csv")]
    if not files:
        print("❌ No submission found!")
        exit(1)
    return f"submissions/{files[0]}"

# 📊 Grade submission
def grade(submission_path):
    submission = pd.read_csv(submission_path)
    gt = pd.read_csv("data/ground_truth.csv")

    # Validate
    if "prediction" not in submission.columns:
        print("❌ Missing 'prediction' column")
        exit(1)

    if len(submission) != len(gt):
        print("❌ Row count mismatch")
        exit(1)

    y_true = gt["label"]
    y_pred = submission["prediction"]

    f1 = round(f1_score(y_true, y_pred, pos_label="HIGH"), 4)
    acc = round((y_true == y_pred).mean() * 100, 2)

    print(f"✅ Accuracy: {acc}%")
    print(f"✅ F1 Score: {f1}")

    # 🔥 Baseline (simple rule for comparison)
    test = pd.read_csv("data/test.csv")
    baseline_pred = test["glucose"].apply(lambda x: "HIGH" if x > 140 else "LOW")
    baseline_f1 = round(f1_score(gt["label"], baseline_pred, pos_label="HIGH"), 4)

    # 📌 Load leaderboard
    lb_file = "leaderboard_data/leaderboard.json"
    if os.path.exists(lb_file):
        with open(lb_file) as f:
            leaderboard = json.load(f)
    else:
        leaderboard = []

    # 🧹 Remove old entry for same submission
    leaderboard = [x for x in leaderboard if x["submission"] != submission_path]

    # ➕ Add submission result
    leaderboard.append({
        "submission": submission_path,
        "accuracy": acc,
        "f1_score": f1,
        "date": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    })

    # ➕ Add baseline (once)
    leaderboard = [x for x in leaderboard if x["submission"] != "BASELINE"]
    leaderboard.append({
        "submission": "BASELINE",
        "accuracy": None,
        "f1_score": baseline_f1,
        "date": "-"
    })

    # 🔝 Sort by F1
    leaderboard = sorted(leaderboard, key=lambda x: x["f1_score"], reverse=True)

    # 💾 Save
    with open(lb_file, "w") as f:
        json.dump(leaderboard, f, indent=2)

    print("📊 Leaderboard updated!")

# 🚀 Run
if __name__ == "__main__":
    submission = find_submission()
    grade(submission)

