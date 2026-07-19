"""
app.py — Flask backend for Student Performance Prediction
=========================================================
Serves the HTML frontend and exposes two API endpoints:
  POST /predict       → returns predicted final_score + tier + advice
  GET  /dataset-stats → returns summary statistics from the CSV for the dashboard
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from flask import Flask, render_template, request, jsonify

# ─────────────────────────────────────────────────────────────
# Flask app setup
# ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "student_performance_model.pkl")
CSV_PATH   = os.path.join(BASE_DIR, "StudentsPerformance.csv")

app = Flask(__name__, template_folder="templates")


# ─────────────────────────────────────────────────────────────
# Load (or auto-train) the model at startup
# ─────────────────────────────────────────────────────────────
def load_or_train_model():
    """
    Try to load the pre-trained pipeline from disk.
    If the .pkl file is missing, train the Random Forest pipeline
    on-the-fly from the CSV and save it for future runs.
    """
    if os.path.exists(MODEL_PATH):
        print(f"[INFO] Loading model from: {MODEL_PATH}")
        return joblib.load(MODEL_PATH)

    print("[INFO] Model not found — training from scratch …")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor

    df = pd.read_csv(CSV_PATH)
    df["final_score"] = (
        (df["math score"] + df["reading score"] + df["writing score"]) / 3
    ).round(2)

    feature_cols = [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
    ]
    X = df[feature_cols]
    y = df["final_score"]

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[("onehot", ohe, feature_cols)], remainder="drop"
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ]
    )
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[INFO] Model trained and saved to: {MODEL_PATH}")
    return pipeline


model = load_or_train_model()


# ─────────────────────────────────────────────────────────────
# Helper: compute dataset statistics once at startup
# ─────────────────────────────────────────────────────────────
def compute_dataset_stats():
    """Return summary statistics from the CSV for the frontend dashboard."""
    df = pd.read_csv(CSV_PATH)
    df["final_score"] = (
        (df["math score"] + df["reading score"] + df["writing score"]) / 3
    ).round(2)

    stats = {
        "total_students": int(len(df)),
        "avg_final_score": round(float(df["final_score"].mean()), 2),
        "max_final_score": round(float(df["final_score"].max()), 2),
        "min_final_score": round(float(df["final_score"].min()), 2),
        # Average score by gender
        "gender_avg": (
            df.groupby("gender")["final_score"]
            .mean()
            .round(2)
            .to_dict()
        ),
        # Average score by lunch type
        "lunch_avg": (
            df.groupby("lunch")["final_score"]
            .mean()
            .round(2)
            .to_dict()
        ),
        # Average score by test prep
        "prep_avg": (
            df.groupby("test preparation course")["final_score"]
            .mean()
            .round(2)
            .to_dict()
        ),
        # Average score by race/ethnicity group
        "race_avg": (
            df.groupby("race/ethnicity")["final_score"]
            .mean()
            .round(2)
            .sort_index()
            .to_dict()
        ),
        # Average score by parental education
        "parental_avg": (
            df.groupby("parental level of education")["final_score"]
            .mean()
            .round(2)
            .to_dict()
        ),
        # Score distribution buckets for histogram
        "score_distribution": (
            pd.cut(
                df["final_score"],
                bins=[0, 40, 50, 60, 70, 80, 90, 101],
                labels=["0-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100"],
                right=False,
            )
            .value_counts()
            .sort_index()
            .to_dict()
        ),
    }
    return stats


DATASET_STATS = compute_dataset_stats()


# ─────────────────────────────────────────────────────────────
# Helper: determine performance tier and advice
# ─────────────────────────────────────────────────────────────
def get_tier_and_advice(score: float) -> dict:
    """Map a predicted score to a performance tier, colour and advice string."""
    if score >= 85:
        return {
            "tier": "Excellent",
            "tier_class": "tier-excellent",
            "emoji": "🌟",
            "color": "#22c55e",
            "advice": (
                "Outstanding performance predicted! This student is on track for top grades. "
                "Encourage advanced coursework and extracurricular academic activities."
            ),
        }
    elif score >= 70:
        return {
            "tier": "Good",
            "tier_class": "tier-good",
            "emoji": "✅",
            "color": "#3b82f6",
            "advice": (
                "Above-average performance expected. A little extra effort in weak areas "
                "could push scores into the excellent range."
            ),
        }
    elif score >= 55:
        return {
            "tier": "Average",
            "tier_class": "tier-average",
            "emoji": "📊",
            "color": "#f59e0b",
            "advice": (
                "Average performance predicted. Completing a test preparation course and "
                "ensuring access to standard lunch resources can significantly help."
            ),
        }
    elif score >= 40:
        return {
            "tier": "Below Average",
            "tier_class": "tier-below",
            "emoji": "⚠️",
            "color": "#f97316",
            "advice": (
                "Below-average performance predicted. Early intervention is recommended — "
                "consider tutoring, structured study plans, and test prep enrollment."
            ),
        }
    else:
        return {
            "tier": "At Risk",
            "tier_class": "tier-risk",
            "emoji": "🆘",
            "color": "#ef4444",
            "advice": (
                "Student may be at serious academic risk. Immediate support, counselling, "
                "and resource access (meals, prep courses) are strongly advised."
            ),
        }


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main HTML frontend."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a JSON body with the 5 demographic fields and returns
    a JSON response containing the predicted final_score and metadata.

    Expected JSON body:
    {
        "gender": "female",
        "race_ethnicity": "group C",
        "parental_education": "bachelor's degree",
        "lunch": "standard",
        "test_prep": "completed"
    }
    """
    try:
        data = request.get_json(force=True)

        # Map frontend field names → model column names
        student = {
            "gender"                      : data.get("gender", ""),
            "race/ethnicity"              : data.get("race_ethnicity", ""),
            "parental level of education" : data.get("parental_education", ""),
            "lunch"                       : data.get("lunch", ""),
            "test preparation course"     : data.get("test_prep", ""),
        }

        # Validate that all fields are present
        for field, value in student.items():
            if not value:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Build DataFrame and predict
        input_df = pd.DataFrame([student])
        predicted = float(model.predict(input_df)[0])
        predicted = max(0.0, min(100.0, round(predicted, 2)))  # clamp to [0, 100]

        tier_info = get_tier_and_advice(predicted)

        # Percentile in dataset
        df_ref = pd.read_csv(CSV_PATH)
        df_ref["final_score"] = (
            (df_ref["math score"] + df_ref["reading score"] + df_ref["writing score"]) / 3
        )
        percentile = round(
            float((df_ref["final_score"] < predicted).mean() * 100), 1
        )

        return jsonify({
            "success"    : True,
            "score"      : predicted,
            "percentile" : percentile,
            "tier"       : tier_info["tier"],
            "tier_class" : tier_info["tier_class"],
            "emoji"      : tier_info["emoji"],
            "color"      : tier_info["color"],
            "advice"     : tier_info["advice"],
            "input"      : student,
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/dataset-stats")
def dataset_stats():
    """Return pre-computed dataset statistics as JSON for the dashboard charts."""
    return jsonify(DATASET_STATS)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Student Performance Prediction — Flask App")
    print("  http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, host="0.0.0.0", port=5000)
