# Student Performance ML Model

A Flask-based machine learning web app that predicts a student's final score (0–100) from demographic and academic background inputs.

## Overview

This project uses the `StudentsPerformance.csv` dataset and a scikit-learn Random Forest pipeline to:

- predict a student's final score
- classify the prediction into performance tiers with guidance
- show interactive dataset insights through a dashboard

## Features

- **Prediction API** (`POST /predict`) using:
  - gender
  - race/ethnicity
  - parental level of education
  - lunch type
  - test preparation course
- **Dataset statistics API** (`GET /dataset-stats`) for dashboard charts
- **Auto model training** if `student_performance_model.pkl` is missing
- **Interactive frontend dashboard** (single-page UI with Chart.js)

## Project Structure

```text
StudentPerformance-ML-Model/
├── app.py
├── StudentsPerformance.csv
├── student_performance_model.pkl
├── student_performance_prediction.ipynb
└── templates/
    └── index.html
```

## Tech Stack

- Python
- Flask
- pandas
- numpy
- scikit-learn
- joblib
- HTML/CSS/JavaScript
- Chart.js

## Setup

1. **Clone the repository**
2. **Create and activate a virtual environment** (recommended)
3. **Install dependencies**:

```bash
pip install flask pandas numpy scikit-learn joblib
```

## Run the App

From the repository root:

```bash
python app.py
```

Then open: `http://127.0.0.1:5000`

## API Endpoints

### `POST /predict`

Request body:

```json
{
  "gender": "female",
  "race_ethnicity": "group C",
  "parental_education": "bachelor's degree",
  "lunch": "standard",
  "test_prep": "completed"
}
```

Response includes:

- predicted `score`
- `percentile`
- `tier`, `tier_class`, `emoji`, `color`
- improvement `advice`

### `GET /dataset-stats`

Returns precomputed dataset metrics used by the dashboard:

- total students
- min/avg/max final score
- averages grouped by gender, lunch, test prep, race, and parental education
- score distribution buckets

## Notes

- If `student_performance_model.pkl` is not found, the app trains a new model from `StudentsPerformance.csv` on startup and saves it automatically.
- The notebook `student_performance_prediction.ipynb` can be used for exploratory analysis/training experiments.
