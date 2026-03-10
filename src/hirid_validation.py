"""
HiRID External Validation
Validates IDI-enhanced mortality model on HiRID (Swiss all-ICU cohort, 2008-2016).

HiRID cohort: 33,897 ICU admissions, 6.08% ICU mortality
Test set:     6,779 admissions (80/20 split)
Key result:   AUROC 0.9063 (95% CI 0.89-0.92), calibration slope 0.98

Reference: Collier & Shalhout, npj Digital Medicine (2026)
HiRID: https://physionet.org/content/hirid/1.1.1/
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
import joblib
import argparse


def load_hirid_cohort(hirid_path: str) -> pd.DataFrame:
    """
    Load and preprocess HiRID cohort.
    Returns DataFrame with stay_id, outcome, and IDI features.
    Note: HiRID documentation latency ~1.2 min vs MIMIC-IV ~15 h.
    """
    # Load general table for demographics and outcomes
    general = pd.read_csv(f"{hirid_path}/general_table.csv")
    # Load observation table for timestamps
    # (actual implementation depends on HiRID file structure)
    # Placeholder — replace with actual HiRID loading logic
    return general


def calibration_slope(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute calibration slope via logistic regression of log-odds.
    Slope = 1.0 is perfect; > 1 = overconfident, < 1 = underconfident.
    """
    from sklearn.linear_model import LogisticRegression
    log_odds = np.log(y_pred / (1 - y_pred + 1e-9))
    lr = LogisticRegression(fit_intercept=True)
    lr.fit(log_odds.reshape(-1, 1), y_true)
    return float(lr.coef_[0][0])


def validate_on_hirid(model_path: str,
                       scaler_path: str,
                       hirid_features_path: str,
                       outcome_col: str = "icu_mortality"):
    """
    Load trained MIMIC-IV model and evaluate on HiRID test set.
    Reports AUROC, AUPRC, Brier, calibration slope.
    """
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    df     = pd.read_csv(hirid_features_path)

    feature_cols = [c for c in df.columns if c.startswith("idi_") or
                    c in ["age", "sex_male"]]
    X = scaler.transform(df[feature_cols].fillna(0))
    y = df[outcome_col].values

    y_pred = model.predict_proba(X)[:, 1]

    auroc  = roc_auc_score(y, y_pred)
    auprc  = average_precision_score(y, y_pred)
    brier  = brier_score_loss(y, y_pred)
    cal_sl = calibration_slope(y, y_pred)

    print("HiRID External Validation Results")
    print("=" * 40)
    print(f"N:                {len(y):,}")
    print(f"ICU mortality:    {y.mean():.2%}")
    print(f"AUROC:            {auroc:.4f}")
    print(f"AUPRC:            {auprc:.4f}")
    print(f"Brier score:      {brier:.4f}")
    print(f"Calibration slope:{cal_sl:.3f}")
    print()
    print("Target results (paper):")
    print("  AUROC  0.9063 (95% CI 0.89-0.92)")
    print("  AUPRC  0.4546")
    print("  Brier  0.1168")
    print("  Cal.   0.98")

    return {"auroc": auroc, "auprc": auprc, "brier": brier, "cal_slope": cal_sl}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True)
    parser.add_argument("--scaler",   required=True)
    parser.add_argument("--features", required=True)
    args = parser.parse_args()
    validate_on_hirid(args.model, args.scaler, args.features)
