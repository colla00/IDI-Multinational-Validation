"""
Model Training — npj Multinational Validation
Baseline (age + sex + ICU LOS) vs IDI-enhanced logistic regression.
Temporal split: train 2008-2018, test 2019.
Results: MIMIC-IV test AUROC 0.6401 (baseline 0.6153)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import argparse
from metrics import full_metrics


BASELINE_FEATURES = ["age", "sex_male", "icu_los_hours"]
IDI_FEATURES = [
    "idi_events_24h", "idi_events_per_hour", "idi_mean_interevent_min",
    "idi_std_interevent_min", "idi_cv_interevent", "idi_max_gap_min",
    "idi_gap_count_60m", "idi_gap_count_120m", "idi_burstiness",
]
OUTCOME = "hospital_mortality"


def temporal_split(df, year_col="admit_year", test_year=2019):
    train = df[df[year_col] < test_year]
    test  = df[df[year_col] == test_year]
    return train, test


def train_and_evaluate(data_path: str, split: str = "temporal",
                       save_model: bool = True):
    df = pd.read_csv(data_path)

    if split == "temporal":
        train, test = temporal_split(df)
        print(f"Temporal split: train n={len(train):,}, test n={len(test):,}")
    else:
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(df, test_size=0.2, random_state=42,
                                        stratify=df[OUTCOME])
        print(f"Random 80/20 split: train n={len(train):,}, test n={len(test):,}")

    y_train = train[OUTCOME].values
    y_test  = test[OUTCOME].values

    print(f"Train mortality: {y_train.mean():.2%}")
    print(f"Test mortality:  {y_test.mean():.2%}")

    scaler = StandardScaler()

    # Baseline model
    X_train_b = scaler.fit_transform(train[BASELINE_FEATURES].fillna(0))
    X_test_b  = scaler.transform(test[BASELINE_FEATURES].fillna(0))
    lr_base   = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_base.fit(X_train_b, y_train)
    y_pred_base = lr_base.predict_proba(X_test_b)[:, 1]

    # IDI-enhanced model
    all_features = BASELINE_FEATURES + IDI_FEATURES
    scaler_idi = StandardScaler()
    X_train_i = scaler_idi.fit_transform(train[all_features].fillna(0))
    X_test_i  = scaler_idi.transform(test[all_features].fillna(0))
    lr_idi    = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_idi.fit(X_train_i, y_train)
    y_pred_idi = lr_idi.predict_proba(X_test_i)[:, 1]

    results = full_metrics(y_test, y_pred_base, y_pred_idi,
                            label=f"MIMIC-IV ({split} split)")

    if save_model:
        joblib.dump(lr_idi,    "results/model_idi.pkl")
        joblib.dump(scaler_idi,"results/scaler_idi.pkl")
        print("\nModels saved to results/")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  required=True)
    parser.add_argument("--split", default="temporal",
                        choices=["temporal","random"])
    args = parser.parse_args()
    train_and_evaluate(args.data, args.split)
