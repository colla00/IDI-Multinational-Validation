"""
Temporal Leakage Filter
Removes IDI features correlated with ICU LOS (|r| > 0.30).
Non-survivors have longer ICU stays; features tracking stay duration
are reverse-causal proxies, not true predictors.

Reference: Collier & Shalhout, npj Digital Medicine (2026)
MIMIC-IV: non-survivors median 71.3 h vs survivors 53.1 h ICU LOS
Feature reduction: 112 candidate → 90 after review → 45 final (45 pass filter)
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
    Returns (filtered_df, kept_cols, dropped_cols_with_r).
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

    return df[kept + [c for c in df.columns if c not in feature_cols]], kept, dropped


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--los_col", default="icu_los_hours")
    parser.add_argument("--threshold", type=float, default=0.30)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    feature_cols = [c for c in df.columns if c.startswith("idi_")]
    df_filtered, kept, dropped = leakage_filter(
        df, feature_cols, args.los_col, args.threshold
    )
    df_filtered.to_csv(args.output, index=False)
    print(f"Saved filtered features to {args.output}")
