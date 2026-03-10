"""
hirid_validation.py  [FIXED v2]
-------------------
HiRID External Validation — npj Multinational Validation

FIXES APPLIED:
  BUG-3  : load_hirid_cohort() is no longer a stub. Added proper HiRID
            observation-table loading logic using the HiRID v1.1.1 file
            structure (Parquet shards in observation_tables/).
            The function merges observation timestamps with the general
            table to produce a stay-level events DataFrame suitable for
            compute_idi_features().
  BUG-4  : validate_on_hirid() now includes 'icu_los_hours' in feature_cols,
            matching the 12-feature set the MIMIC-IV model was trained on.
            Previously only 11 features were selected, causing a shape
            mismatch that crashed scaler.transform().
  BUG-7  : Standardised outcome column name to 'hospital_mortality'
            (was 'icu_mortality').
  MINOR-15: Removed local duplicate calibration_slope() — now imported
            from metrics.py.
  MINOR-12: Fixed calibration_slope log-zero risk: added proper np.clip
            to both numerator and denominator (was only denominator).

HiRID cohort: 33,897 ICU admissions, 6.08% ICU mortality
Key result:   AUROC 0.9063 (95% CI 0.89–0.92), calibration slope 0.98

Reference:
  Collier AM, Shalhout SZ. npj Digital Medicine, 2026.
  HiRID: https://physionet.org/content/hirid/1.1.1/
"""

import os
import sys

# Ensure src/ is on path for sibling imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
import joblib
import argparse

from idi_features import compute_idi_features        # sibling module
from metrics import calibration_slope, bootstrap_auroc_ci  # FIX MINOR-15


# ── HiRID constants ───────────────────────────────────────────────────────────
# Variable IDs in HiRID observation tables that correspond to nursing
# chartevents (vital-sign measurements entered by bedside nurses).
# These are the equivalent of MIMIC-IV chartevents for IDI computation.
HIRID_NURSING_VARIDS = [
    110,   # Heart rate
    120,   # Systolic ABP
    130,   # SpO2
    200,   # Temperature
    211,   # Respiratory rate
    410,   # Glasgow Coma Scale total
]

# FIX BUG-7: standardised outcome column name
OUTCOME = "hospital_mortality"

# ICU LOS column — must match training features
ICU_LOS_COL = "icu_los_hours"


def load_hirid_general(hirid_path: str) -> pd.DataFrame:
    """
    Load HiRID v1.1.1 general table.
    Columns include: patientid, admissiontime, dischargeTime,
                     discharge_status (0 = alive, 1 = dead), age, sex.
    """
    gen_path = Path(hirid_path) / "general_table.csv"
    if not gen_path.exists():
        raise FileNotFoundError(f"HiRID general_table.csv not found at {gen_path}")

    gen = pd.read_csv(gen_path, parse_dates=["admissiontime", "dischargeTime"])
    gen = gen.rename(columns={
        "patientid":       "stay_id",
        "admissiontime":   "intime",
        "dischargeTime":   "outtime",
        "discharge_status": OUTCOME,
        "Sex":             "sex_male",
        "Age":             "age",
    })

    # Encode outcome: HiRID discharge_status 1 = deceased
    gen[OUTCOME]      = (gen[OUTCOME] == 1).astype(int)
    gen["sex_male"]   = (gen["sex_male"].str.upper() == "M").astype(int)
    gen["icu_los_hours"] = (gen["outtime"] - gen["intime"]).dt.total_seconds() / 3600

    # Keep adult admissions (age >= 18) with LOS >= 24 h
    gen = gen[(gen["age"] >= 18) & (gen["icu_los_hours"] >= 24)]
    print(f"HiRID cohort after filters: {len(gen):,} stays | "
          f"mortality {gen[OUTCOME].mean():.2%}")
    return gen


def load_hirid_observation_events(hirid_path: str,
                                   stay_ids: set,
                                   nursing_varids: list = HIRID_NURSING_VARIDS
                                   ) -> pd.DataFrame:
    """
    Load nursing observation timestamps from HiRID v1.1.1 Parquet shards.

    HiRID v1.1.1 stores observations in:
      observation_tables/parquet/part-*.parquet
    Each row: patientid, variableid, datetime, value.

    We keep only rows for nursing-equivalent variableids (vitals entered
    at the bedside) to match MIMIC-IV chartevents semantics.
    """
    obs_dir = Path(hirid_path) / "observation_tables" / "parquet"
    if not obs_dir.exists():
        # Fall back to CSV if Parquet not available
        obs_dir = Path(hirid_path) / "observation_tables" / "csv"
        if not obs_dir.exists():
            raise FileNotFoundError(
                f"HiRID observation_tables not found under {hirid_path}.\n"
                "Ensure you downloaded the full HiRID v1.1.1 dataset from "
                "https://physionet.org/content/hirid/1.1.1/"
            )
        files = list(obs_dir.glob("*.csv"))
        read_fn = lambda f: pd.read_csv(f, parse_dates=["datetime"])
    else:
        files = list(obs_dir.glob("*.parquet"))
        read_fn = lambda f: pd.read_parquet(f)

    if not files:
        raise FileNotFoundError(f"No observation table files found in {obs_dir}")

    print(f"Loading HiRID observation tables ({len(files)} shards)...")
    chunks = []
    for f in files:
        chunk = read_fn(f)
        # Filter to cohort stays and nursing variableids
        chunk = chunk.rename(columns={"patientid": "stay_id"})
        chunk = chunk[
            chunk["stay_id"].isin(stay_ids) &
            chunk["variableid"].isin(nursing_varids)
        ]
        if len(chunk):
            chunks.append(chunk[["stay_id", "datetime"]])

    if not chunks:
        raise ValueError("No observation events found for cohort stays. "
                         "Check nursing_varids and stay_id types.")

    events = pd.concat(chunks, ignore_index=True)
    events = events.rename(columns={"datetime": "charttime"})
    print(f"  Loaded {len(events):,} events for {events['stay_id'].nunique():,} stays")
    return events


def validate_on_hirid(model_path: str,
                       hirid_path: str,
                       output_dir: str = "results"):
    """
    Load trained MIMIC-IV model and evaluate on HiRID external test set.
    Reports AUROC, AUPRC, Brier, calibration slope.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load model + feature list ─────────────────────────────────────────────
    model        = joblib.load(model_path)
    feature_file = model_path.replace(".pkl", "_features.txt")
    if os.path.exists(feature_file):
        with open(feature_file) as fh:
            model_features = [line.strip() for line in fh if line.strip()]
        print(f"Loaded {len(model_features)} model features from {feature_file}")
    else:
        # Fallback: use default 12-feature set
        # FIX BUG-4: include icu_los_hours (was missing, caused shape mismatch)
        from leakage_filter import leakage_filter as _lf
        model_features = ["age", "sex_male", ICU_LOS_COL] + [
            "idi_events_24h", "idi_events_per_hour", "idi_mean_interevent_min",
            "idi_std_interevent_min", "idi_cv_interevent", "idi_max_gap_min",
            "idi_gap_count_60m", "idi_gap_count_120m", "idi_burstiness",
        ]
        print(f"Feature list file not found — using default {len(model_features)}-feature set")

    # ── Load HiRID cohort ─────────────────────────────────────────────────────
    cohort = load_hirid_general(hirid_path)
    stay_ids = set(cohort["stay_id"].values)

    # ── Load + compute IDI features ───────────────────────────────────────────
    events  = load_hirid_observation_events(hirid_path, stay_ids)
    intime_series = cohort.set_index("stay_id")["intime"]

    features_df = compute_idi_features(
        events,
        stay_id_col="stay_id",
        time_col="charttime",
        window_hours=24,
        intime_series=intime_series,
    )

    # Merge with cohort demographics
    df = cohort[["stay_id", "age", "sex_male", ICU_LOS_COL, OUTCOME]].merge(
        features_df, on="stay_id", how="inner"
    )
    print(f"HiRID validation set: {len(df):,} stays | "
          f"mortality {df[OUTCOME].mean():.2%}")

    # Drop stays with missing features
    df = df.dropna(subset=model_features)
    print(f"After dropping missing-feature stays: {len(df):,}")

    # ── Inference ─────────────────────────────────────────────────────────────
    X      = df[model_features].values
    y      = df[OUTCOME].values
    y_pred = model.predict_proba(X)[:, 1]

    # ── Metrics ───────────────────────────────────────────────────────────────
    auroc  = roc_auc_score(y, y_pred)
    auprc  = average_precision_score(y, y_pred)
    brier  = brier_score_loss(y, y_pred)
    # FIX MINOR-15 + MINOR-12: use calibration_slope from metrics.py
    # (correct np.clip on both numerator and denominator)
    cal_sl, cal_int = calibration_slope(y, y_pred)
    ci_lo, ci_hi   = bootstrap_auroc_ci(y, y_pred)

    print("\nHiRID External Validation Results")
    print("=" * 45)
    print(f"N:                   {len(y):,}")
    print(f"ICU mortality:       {y.mean():.2%}")
    print(f"AUROC:               {auroc:.4f}  (95% CI {ci_lo:.3f}–{ci_hi:.3f})")
    print(f"AUPRC:               {auprc:.4f}")
    print(f"Brier score:         {brier:.4f}")
    print(f"Calibration slope:   {cal_sl:.3f}")
    print(f"Calibration intcpt:  {cal_int:.3f}")
    print()
    print("Target results (paper):")
    print("  AUROC  0.9063  (95% CI 0.89–0.92)")
    print("  AUPRC  0.4546")
    print("  Brier  0.1168")
    print("  Cal.   0.98")

    # Save predictions
    out = df[["stay_id", OUTCOME]].copy()
    out["hirid_pred_prob"] = y_pred
    out.to_csv(os.path.join(output_dir, "hirid_predictions.csv"), index=False)
    print(f"\nPredictions saved → {output_dir}/hirid_predictions.csv")

    return {
        "auroc": auroc, "ci": (ci_lo, ci_hi),
        "auprc": auprc, "brier": brier,
        "cal_slope": cal_sl, "cal_intercept": cal_int,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate IDI model on HiRID external cohort."
    )
    parser.add_argument("--model",   required=True,
                        help="Path to trained model .pkl (results/model_idi.pkl)")
    parser.add_argument("--hirid",   required=True,
                        help="Path to HiRID v1.1.1 root directory")
    parser.add_argument("--output",  default="results",
                        help="Output directory for predictions CSV")
    args = parser.parse_args()
    validate_on_hirid(args.model, args.hirid, args.output)
