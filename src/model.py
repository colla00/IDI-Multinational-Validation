"""
model.py  [FIXED v2]
--------
Model Training — npj Multinational Validation

FIXES APPLIED:
  BUG-5 : Added sys.path insertion so `from metrics import full_metrics`
          works when called with `python src/model.py` from project root.
  BUG-6 : Added os.makedirs("results", exist_ok=True) before joblib.dump
          calls — prevents FileNotFoundError on first run.
  BUG-7 : Standardised outcome column name to 'hospital_mortality'
          (was 'hospital_mortality' here but 'icu_mortality' in
          hirid_validation.py — now consistent across all scripts).
  ISSUE-11: Replaced fillna(0) with training-set median imputation.
            Zero-imputation for icu_los_hours implied a 0-hour ICU stay,
            which is clinically nonsensical and distorts the model.

Baseline: age, sex_male, icu_los_hours
IDI-enhanced: baseline + IDI features surviving leakage filter
Temporal split: train 2008-2018, test 2019.
"""

import os
import sys

# FIX BUG-5: ensure src/ is on the path regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import argparse

from metrics import full_metrics   # now importable from any working directory


BASELINE_FEATURES = ["age", "sex_male", "icu_los_hours"]
IDI_FEATURES = [
    "idi_events_24h", "idi_events_per_hour", "idi_mean_interevent_min",
    "idi_std_interevent_min", "idi_cv_interevent", "idi_max_gap_min",
    "idi_gap_count_60m", "idi_gap_count_120m", "idi_burstiness",
]

# FIX BUG-7: standardised outcome column name
OUTCOME = "hospital_mortality"


# ── Leakage filter ────────────────────────────────────────────────────────────
def apply_leakage_filter(train_df: pd.DataFrame,
                          feature_cols: list,
                          los_col: str = "icu_los_hours",
                          threshold: float = 0.30) -> list:
    """Return IDI feature names with |r| <= threshold vs ICU LOS."""
    kept, dropped = [], []
    for col in feature_cols:
        valid = train_df[[col, los_col]].dropna()
        if len(valid) < 10:
            kept.append(col)
            continue
        r = valid.corr().iloc[0, 1]
        if abs(r) <= threshold:
            kept.append(col)
        else:
            dropped.append((col, round(r, 4)))
    if dropped:
        print(f"Leakage filter dropped {len(dropped)} features:")
        for c, r in sorted(dropped, key=lambda x: abs(x[1]), reverse=True):
            print(f"  {c}: r={r}")
    print(f"IDI features retained: {len(kept)} / {len(feature_cols)}")
    return kept


# ── Pipeline builder (scaler + median imputer + logistic regression) ──────────
def build_pipeline() -> Pipeline:
    return Pipeline([
        # FIX ISSUE-11: median imputation instead of fillna(0)
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])


# ── Train / test split ────────────────────────────────────────────────────────
def temporal_split(df: pd.DataFrame, year_col: str = "admit_year",
                   test_year: int = 2019):
    train = df[df[year_col] <  test_year]
    test  = df[df[year_col] == test_year]
    return train, test


# ── Main training function ────────────────────────────────────────────────────
def train_and_evaluate(data_path: str, split: str = "temporal",
                       save_model: bool = True):
    df = pd.read_csv(data_path, parse_dates=["intime"])
    df["admit_year"] = df["intime"].dt.year
    df["sex_male"]   = (df["gender"].str.upper() == "M").astype(int)

    if split == "temporal":
        train, test = temporal_split(df)
        print(f"Temporal split | train n={len(train):,} | test n={len(test):,}")
    else:
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(df, test_size=0.2, random_state=42,
                                        stratify=df[OUTCOME])
        print(f"Random 80/20   | train n={len(train):,} | test n={len(test):,}")

    print(f"Train mortality: {train[OUTCOME].mean():.2%}")
    print(f"Test  mortality: {test[OUTCOME].mean():.2%}")

    y_train = train[OUTCOME].values
    y_test  = test[OUTCOME].values

    # Leakage filter on training set only
    idi_kept = apply_leakage_filter(train, IDI_FEATURES)

    # ── Baseline model ────────────────────────────────────────────────────────
    pipe_base = build_pipeline()
    pipe_base.fit(train[BASELINE_FEATURES], y_train)
    y_pred_base = pipe_base.predict_proba(test[BASELINE_FEATURES])[:, 1]

    # ── IDI-enhanced model ────────────────────────────────────────────────────
    all_features = BASELINE_FEATURES + idi_kept
    pipe_idi = build_pipeline()
    pipe_idi.fit(train[all_features], y_train)
    y_pred_idi = pipe_idi.predict_proba(test[all_features])[:, 1]

    results = full_metrics(y_test, y_pred_base, y_pred_idi,
                            label=f"MIMIC-IV ({split} split)")

    # ── FIX BUG-6: ensure results/ exists before saving ──────────────────────
    os.makedirs("results", exist_ok=True)

    if save_model:
        joblib.dump(pipe_idi,  "results/model_idi.pkl")
        # Save feature list alongside model for reproducibility
        pd.Series(all_features).to_csv("results/model_idi_features.txt",
                                        index=False, header=False)
        print("Models saved → results/model_idi.pkl")
        print("Feature list → results/model_idi_features.txt")

    # Save combined predictions for downstream analysis
    pred_df = test[["stay_id", OUTCOME]].copy()
    pred_df["baseline_prob"] = y_pred_base
    pred_df["idi_prob"]      = y_pred_idi
    pred_df.to_csv("results/test_predictions.csv", index=False)
    print("Predictions  → results/test_predictions.csv")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  required=True,
                        help="Path to IDI features CSV (output of idi_features.py)")
    parser.add_argument("--split", default="temporal",
                        choices=["temporal", "random"])
    args = parser.parse_args()
    train_and_evaluate(args.data, args.split)
