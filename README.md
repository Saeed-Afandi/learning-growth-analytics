# Learning Growth Analytics

Python data science and machine learning project exploring academic learning growth and student performance.

## Overview

This project explores how data can be used to understand learning progress and support better study decisions. It starts with a small anonymized longitudinal learning dataset and extends the analysis to a public student-performance dataset from the UCI Machine Learning Repository.

The project combines Python, data analysis, visualization, and introductory machine learning while keeping the interpretation cautious and privacy-aware.

## Part 1 — Learning Progress Analysis

`src/progress_analysis.py` analyzes four assessment periods of anonymized RIT-score data using pandas and matplotlib.

The script:

- loads and cleans the dataset;
- calculates score gains across assessment periods;
- compares achievement percentiles between Spring and Fall 2025;
- exports a summary CSV;
- generates visualizations of RIT-score growth and percentile change.

### Observed learning-growth data

| Measure | Earlier value | Later value | Change |
| --- | ---: | ---: | ---: |
| Mathematics RIT | 158 | 182 | +24 |
| Reading RIT | 154 | 168 | +14 |
| Mathematics achievement percentile | 49th | 72nd | +23 percentile points |
| Reading achievement percentile | 33rd | 45th | +12 percentile points |

The strongest measured change is in mathematics, where the RIT score increased by 24 points across the observed period and the achievement percentile rose from the 49th to the 72nd percentile between Spring and Fall 2025.

## Part 2 — Machine Learning Extension

`src/ml_exploration.py` uses the public **UCI Student Performance** dataset (dataset ID 320) to compare two regression approaches for predicting final academic performance:

- Linear Regression
- Random Forest Regression

The model intentionally focuses on study and support variables such as study time, absences, school support, family support, extracurricular participation, internet access, and prior failures. Protected demographic attributes are excluded from the feature set.

The script:

- retrieves the public UCI dataset;
- prepares selected features;
- creates train/test sets;
- trains and compares two regression models;
- reports MAE and R²;
- exports Random Forest feature importance.

This is an educational machine-learning exercise. Associations and feature importance do **not** prove causation, and the model should not be used for high-stakes decisions about students.

## Project Structure

```text
learning-growth-analytics/
├── data/
│   └── progress_data.csv
├── output/
│   └── growth_summary.csv
├── src/
│   ├── progress_analysis.py
│   └── ml_exploration.py
├── .gitignore
├── README.md
└── requirements.txt
```

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/progress_analysis.py
python3 src/ml_exploration.py
```

On Windows:

```text
.venv\\Scripts\\activate
```

The machine-learning script requires an internet connection the first time it retrieves the UCI dataset.

## Tools and Libraries

- Python
- pandas
- matplotlib
- scikit-learn
- ucimlrepo

## Data and Privacy

The longitudinal learning-progress dataset in this repository is anonymized and contains only assessment-period metrics used for this educational analysis. The machine-learning extension uses the public UCI Student Performance dataset.

## Limitations

The learning-progress section contains a small number of observations and should be interpreted descriptively rather than as evidence of causation. The machine-learning section is an educational comparison of models on historical data and is not intended to evaluate individual students or make educational decisions.
