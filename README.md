# 🏥 Diabetes Risk Prediction Leaderboard

A GitHub-based automated system that evaluates diabetes risk predictions and maintains a live leaderboard using GitHub Actions.
the link for leaderboard: https://jd-aniruddha.github.io/cyberbullying-leaderboard/leaderboard.html
---

## 🚀 Project Overview

This project simulates a **machine learning competition platform** where:

* Users submit predictions as CSV files
* The system automatically evaluates them
* Results are ranked on a leaderboard

---

## 🧠 Problem Statement

Given patient health data (age, BMI, glucose, blood pressure), predict whether the patient has:

* **HIGH** diabetes risk
* **LOW** diabetes risk

---

## 📁 Project Structure

```
.
├── data/
│   ├── test.csv
│   └── ground_truth.csv
│
├── submissions/
│   ├── user1.csv
│   ├── user2.csv
│
├── leaderboard.json
├── leaderboard.html
├── grader.py
│
└── .github/workflows/
    └── grader.yml
```

---

## ⚙️ How It Works

```
User Submission → GitHub Action Trigger → Python Grader → Leaderboard Update
```

1. User uploads a CSV file in `submissions/`
2. GitHub Action automatically runs
3. `grader.py` evaluates predictions
4. Accuracy & F1 score are calculated
5. Leaderboard is updated

---

## 📊 Evaluation Metrics

* **Accuracy**
* **F1 Score** (for HIGH risk class)

---

## 📥 Submission Format

CSV file must follow this format:

```
id,prediction
1,HIGH
2,LOW
3,HIGH
```

---

## 📈 Leaderboard

The leaderboard is automatically generated and updated.

👉 View here:
`leaderboard.html` (via GitHub Pages)

---

## 🛠 Tech Stack

* Python (pandas, scikit-learn)
* GitHub Actions (automation)
* HTML + JavaScript (visual leaderboard)
* JSON (data storage)

---

## 🔥 Features

* Automated grading system
* Real-time leaderboard updates
* F1-score based ranking
* Web-based leaderboard UI

---

## ⚠️ Limitations

* Ground truth is public (can be hidden in advanced version)
* No user authentication
* Basic dataset (can be replaced with real-world data)

---

## 🚀 Future Improvements

* Integrate real ML model
* Add private test dataset
* PR-based submissions
* User authentication
* Advanced leaderboard UI

---

## 👨‍💻 Author

Aniruddha Jadhav

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
