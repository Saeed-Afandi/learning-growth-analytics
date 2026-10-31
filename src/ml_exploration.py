"""Exploratory machine-learning exercise using a public education dataset.

This script is for learning model-building concepts only. It is not intended
to evaluate individual students or support admissions, grading, placement,
or other educational decisions.
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

dataset = fetch_ucirepo(id=320)
X_all = dataset.data.features.copy()
y_all = dataset.data.targets.copy()

feature_cols = [
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "internet",
    "absences",
]

X = X_all[feature_cols].copy()
for col in ["schoolsup", "famsup", "paid", "activities", "internet"]:
    X[col] = X[col].map({"yes": 1, "no": 0})

y = pd.to_numeric(y_all["G3"], errors="coerce")
valid = X.notna().all(axis=1) & y.notna()
X = X.loc[valid]
y = y.loc[valid]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=3,
    ),
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rows.append(
        {
            "model": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "R2": r2_score(y_test, predictions),
        }
    )

results = pd.DataFrame(rows).sort_values("MAE")
print(results.to_string(index=False))
results.to_csv(OUT / "ml_model_results.csv", index=False)

print(
    "\nEducational use only: model results describe patterns in a historical "
    "public dataset and should not be used to judge individual students."
)
