"""
leakage_filter.py
-----------------
Removes IDI features correlated with ICU length of stay (|r| > 0.30).

Non-survivors have longer ICU stays (median 71.3 h vs 53.1 h in MIMIC-IV),
so features that track stay duration are reverse-causal proxies rather than
true predictors of mortality. Removing them before model training prevents
this leakage from inflating performance estimates.

In the IDI feature set, idi_events_24h and idi_events_per_hour are typically
excluded by this filter. The remaining 7 features pass.

This module can be used standalone via the command line or imported directly
by model.py and hirid_validation.py.

Reference:
  Collier AM, Shalhout SZ. Intensive Documentation Index as an All-Cause
  Mortality Predictor in Critically Ill Patients: A Multi-Center External
  Validation Study. npj Digital Medicine, 2026.
"""

import pandas as pd
import numpy as np
import argparse


def leakage_filter(df: pd.DataFrame,
                   feature_cols: list,
                   los_col: str = "icu_los_hours",
                   threshold: float = 0.30,
                   verbose: bool = True) -> tuple:
    """
    Remove features with |Pearson r| > threshold with ICU LOS.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature columns and the LOS column.
    feature_cols : list
        Names of candidate feature columns to evaluate.
    los_col : str
        Name of the ICU length-of-stay column.
    threshold : float
        Absolute Pearson r threshold. Features above this are dropped.
    verbose : bool
        If True, print a summary of kept and dropped features.

    Returns
    -------
    filtered_df : pd.DataFrame
        Input DataFrame with dropped feature columns removed.
    kept_cols : list
        Feature names that passed the filter.
    dropped_cols : list of (str, float)
        Feature names and their r values for features that were removed.
    """
    kept, dropped = [], []
    for col in feature_cols:
        valid = df[[col, los_col]].dropna()
        if len(valid) < 10:
            kept.append(col)
            continue
        r = valid.corr().iloc[0, 1]
        if abs(r) <= threshold:
            kept.append(col)
        else:
            dropped.append((col, round(r, 4)))

    if verbose:
        print(f"Leakage filter (|r| > {threshold} with {los_col}):")
        print(f"  Input features:  {len(feature_cols)}")
        print(f"  Kept:            {len(kept)}")
        print(f"  Dropped:         {len(dropped)}")
        for col, r in sorted(dropped, key=lambda x: abs(x[1]), reverse=True):
            print(f"    {col}: r = {r}")

    drop_names = [d[0] for d in dropped]
    non_feature_cols = [c for c in df.columns if c not in feature_cols]
    filtered_df = df[non_feature_cols + kept]
    return filtered_df, kept, dropped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove IDI features correlated with ICU LOS."
    )
    parser.add_argument("--input",     required=True,
                        help="Path to input CSV with IDI features and LOS column")
    parser.add_argument("--output",    required=True,
                        help="Path to save filtered output CSV")
    parser.add_argument("--los_col",   default="icu_los_hours",
                        help="Name of the ICU LOS column (default: icu_los_hours)")
    parser.add_argument("--threshold", type=float, default=0.30,
                        help="Absolute Pearson r threshold (default: 0.30)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    feature_cols = [c for c in df.columns if c.startswith("idi_")]
    df_filtered, kept, dropped = leakage_filter(
        df, feature_cols, args.los_col, args.threshold
    )
    df_filtered.to_csv(args.output, index=False)
    print(f"Saved filtered features to {args.output}")
