# Titanic Survival Prediction

This assignment was for my Artificial Intelligence class (CSCI 431) at Stetson University. The task was to predict which passengers survived the Titanic disaster based on the provided passenger data.

## Problem Type

- **Supervised** — labels are known (`Survived` column: 0 = died, 1 = survived)
- **Binary classification** — two possible outcomes
- **Batch learning** — trained on the full dataset at once

## Data Exploration

Initial findings from `df.describe()` and a few quick charts:

- ~38% of passengers survived overall
- Average passenger age was ~29; most passengers fell in the 20–30 range
- Fares ranged from $0 to $512
- Females survived at a noticeably higher rate than males

## Data Cleaning & Feature Engineering

- Filled missing ages with the median age across all passengers
- Dropped low-signal / high-cardinality columns: `Name`, `Cabin`, `Ticket`
- Encoded categorical columns (`Sex`, `Embarked`) as numeric values
- Added engineered features: `FamilySize` and `IsAlone`

## Models

| Model                | Accuracy | Notes |
|----------------------|----------|-------|
| Logistic Regression  | 78%      | Baseline, no feature engineering |
| Random Forest        | **82%**  | With `FamilySize` / `IsAlone` and tuning |

Random Forest was chosen over a single decision tree because it averages over many trees, which generalizes better than drawing a single decision boundary.

## Files

- [assignment1_titanic.ipynb](assignment1_titanic.ipynb) — full notebook (EDA, cleaning, training, evaluation)
- [assignment1_report.docx](assignment1_report.docx) — written report
- [train.csv](train.csv) / [test.csv](test.csv) — Kaggle Titanic dataset
