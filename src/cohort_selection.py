"""
cohort_selection.py  [NEW - v1]
-------------------
Builds both cohorts for the multinational validation study:
  1. MIMIC-IV derivation cohort  (heart failure, USA)
  2. HiRID external validation cohort (all-ICU, Switzerland)

THIS FILE WAS MISSING FROM THE REPOSITORY.
Added to match the Usage section in README.md:
  python src/cohort_selection.py --mimic data/raw/mimic-iv \
                                  --hirid data/raw/hirid

Outputs (written to data/processed/):
  mimic_cohort.csv  -- 26,153 HF ICU admissions, MIMIC-IV v2.2
  hirid_cohort.csv  -- 33,897 all-ICU admissions, HiRID v1.1.1

Inclusion criteria - MIMIC-IV:
  - Adult ICU admissions (age >= 18)
  - Heart failure as primary or secondary diagnosis (ICD-9/ICD-10)
  - ICU LOS >= 24 hours
  - First ICU admission per patient only

Inclusion criteria - HiRID:
  - Adult ICU admissions (age >= 18)
  - ICU LOS >= 24 hours
  (HiRID is all-ICU; no disease-specific filter applied for external validation)

Outcome variable: hospital_mortality (1 = died in hospital, 0 = survived)

Reference:
  Collier AM, Shalhout SZ.
  Intensive Documentation Index as an All-Cause Mortality Predictor
  in Critically Ill Patients: A Multi-Center External Validation Study.
  Journal of Biomedical Informatics (under review, 2026).
  HiRID dataset: https://physionet.org/content/hirid/1.1.1/
  MIMIC-IV v2.2:  https://physionet.org/content/mimiciv/2.2/
"""

import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path("data/processed")

# ── ICD codes for heart failure ───────────────────────────────────────────────
HF_ICD9 = [
    "4280", "4281", "42820", "42821", "42822", "42823",
    "42830", "42831", "42832", "42833",
    "42840", "42841", "42842", "42843", "4289",
]
HF_ICD10_PREFIXES = [
    "I50", "I500", "I501",
    "I5020", "I5021", "I5022", "I5023",
    "I5030", "I5031", "I5032", "I5033",
    "I5040", "I5041", "I5042", "I5043",
    "I509",  "I5081", "I5082", "I5083", "I5084", "I5089",
]

# MIMIC-IV race -> compact label (substring match, uppercase)
RACE_MAP = {
    "WHITE":    "White",
    "BLACK":    "Black",
    "HISPANIC": "Hispanic",
    "ASIAN":    "Asian",
}

# Standardised outcome column name used across all scripts
OUTCOME = "hospital_mortality"


# ── Shared helpers ────────────────────────────────────────────────────────────

def standardise_race(series: pd.Series) -> pd.Series:
    """Map MIMIC-IV verbose race strings to compact labels; unmatched -> 'Other'."""
    s      = series.fillna("UNKNOWN").str.upper()
    result = pd.Series("Other", index=series.index)
    for keyword, label in RACE_MAP.items():
        result[s.str.contains(keyword, na=False)] = label
    return result


# ── MIMIC-IV cohort ───────────────────────────────────────────────────────────

def _flag_heart_failure(diagnoses: pd.DataFrame) -> set:
    """Return set of hadm_ids that carry any HF ICD-9 or ICD-10 code."""
    icd9_mask  = (diagnoses["icd_version"] == 9) & (
        diagnoses["icd_code"].isin(HF_ICD9))
    icd10_mask = (diagnoses["icd_version"] == 10) & (
        diagnoses["icd_code"].str.startswith(tuple(HF_ICD10_PREFIXES)))
    return set(diagnoses.loc[icd9_mask | icd10_mask, "hadm_id"].unique())


def _compute_anchor_age(df: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """Add 'age' column using MIMIC-IV anchor_age / anchor_year methodology."""
    df = df.merge(
        patients[["subject_id", "anchor_age", "anchor_year", "gender"]],
        on="subject_id", how="left",
    )
    df["admit_year"] = df["admittime"].dt.year
    df["age"]        = df["anchor_age"] + (df["admit_year"] - df["anchor_year"])
    return df


def build_mimic_cohort(mimic_path: str) -> pd.DataFrame:
    """
    Build the MIMIC-IV heart-failure ICU cohort.

    Parameters
    ----------
    mimic_path : str
        Root directory of the MIMIC-IV v2.2 download.
        Must contain hosp/ and icu/ subdirectories.

    Returns
    -------
    pd.DataFrame with columns:
        subject_id, hadm_id, stay_id, intime, outtime,
        admit_year, age, gender, race, icu_los_hours, hospital_mortality
    """
    p = Path(mimic_path)
    print("\n-- Building MIMIC-IV cohort ---------------------------------------")

    admissions = pd.read_csv(
        p / "hosp" / "admissions.csv",
        parse_dates=["admittime", "dischtime", "deathtime"],
        usecols=["hadm_id", "admittime", "dischtime", "deathtime",
                 "hospital_expire_flag", "race"],
    )
    patients  = pd.read_csv(p / "hosp" / "patients.csv")
    diagnoses = pd.read_csv(p / "hosp" / "diagnoses_icd.csv")
    icustays  = pd.read_csv(
        p / "icu" / "icustays.csv",
        parse_dates=["intime", "outtime"],
    )
    print(f"  Raw ICU stays  : {len(icustays):,}")
    print(f"  Raw admissions : {len(admissions):,}")

    # Step 1: merge ICU stays with admissions
    # Exclude subject_id from admissions to prevent _x/_y collision (BUG-1)
    df = icustays.merge(
        admissions[["hadm_id", "admittime", "dischtime", "deathtime",
                    "hospital_expire_flag", "race"]],
        on="hadm_id", how="inner",
    )

    # Step 2: add age and gender
    df = _compute_anchor_age(df, patients)

    # Step 3: adults only
    df = df[df["age"] >= 18]
    print(f"  After age >= 18       : {len(df):,}")

    # Step 4: heart failure diagnosis
    hf_hadms = _flag_heart_failure(diagnoses)
    df = df[df["hadm_id"].isin(hf_hadms)]
    print(f"  After HF filter       : {len(df):,}")

    # Step 5: ICU LOS >= 24 h
    df["icu_los_hours"] = (
        (df["outtime"] - df["intime"]).dt.total_seconds() / 3600
    )
    df = df[df["icu_los_hours"] >= 24]
    print(f"  After LOS >= 24h      : {len(df):,}")

    # Step 6: first ICU admission per patient
    df = df.sort_values(["subject_id", "intime"])
    df = df.groupby("subject_id", as_index=False).first()
    print(f"  After first admission : {len(df):,}")

    # Step 7: outcome + race
    df[OUTCOME] = df["hospital_expire_flag"].astype(int)
    df["race"]  = standardise_race(df["race"])

    # Step 8: drop rows missing key covariates
    df = df.dropna(subset=["age", "icu_los_hours"])

    print(f"\n  MIMIC-IV final cohort : {len(df):,} stays")
    print(f"  In-hospital mortality : "
          f"{df[OUTCOME].mean() * 100:.2f}%  (n={df[OUTCOME].sum():,})")
    print(f"  Race distribution:\n{df['race'].value_counts().to_string()}")

    cols = ["subject_id", "hadm_id", "stay_id", "intime", "outtime",
            "admit_year", "age", "gender", "race",
            "icu_los_hours", OUTCOME]
    return df[cols]


# ── HiRID cohort ──────────────────────────────────────────────────────────────

def build_hirid_cohort(hirid_path: str) -> pd.DataFrame:
    """
    Build the HiRID v1.1.1 all-ICU external validation cohort.

    Parameters
    ----------
    hirid_path : str
        Root directory of the HiRID v1.1.1 download.
        Must contain general_table.csv.

    Returns
    -------
    pd.DataFrame with columns:
        stay_id, intime, outtime, age, sex_male,
        icu_los_hours, hospital_mortality
    """
    p = Path(hirid_path)
    print("\n-- Building HiRID cohort ------------------------------------------")

    gen_path = p / "general_table.csv"
    if not gen_path.exists():
        raise FileNotFoundError(
            f"HiRID general_table.csv not found at {gen_path}.\n"
            "Download HiRID v1.1.1 from: "
            "https://physionet.org/content/hirid/1.1.1/"
        )

    gen = pd.read_csv(
        gen_path,
        parse_dates=["admissiontime", "dischargeTime"],
    )
    print(f"  Raw HiRID records: {len(gen):,}")

    gen = gen.rename(columns={
        "patientid":        "stay_id",
        "admissiontime":    "intime",
        "dischargeTime":    "outtime",
        "discharge_status": OUTCOME,
        "Sex":              "sex_male",
        "Age":              "age",
    })

    # Encode outcome: HiRID discharge_status == 1 means deceased
    gen[OUTCOME]    = (gen[OUTCOME] == 1).astype(int)
    gen["sex_male"] = (gen["sex_male"].str.upper() == "M").astype(int)

    gen["icu_los_hours"] = (
        (gen["outtime"] - gen["intime"]).dt.total_seconds() / 3600
    )

    # Adults (age >= 18) with ICU LOS >= 24 h
    gen = gen[(gen["age"] >= 18) & (gen["icu_los_hours"] >= 24)]
    gen = gen.dropna(subset=["age", "icu_los_hours"])

    print(f"  HiRID final cohort : {len(gen):,} stays")
    print(f"  ICU mortality      : "
          f"{gen[OUTCOME].mean() * 100:.2f}%  (n={gen[OUTCOME].sum():,})")

    cols = ["stay_id", "intime", "outtime", "age", "sex_male",
            "icu_los_hours", OUTCOME]
    return gen[cols]


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build MIMIC-IV and HiRID cohorts for the multinational "
            "IDI validation study."
        )
    )
    parser.add_argument(
        "--mimic", required=True,
        help="Path to MIMIC-IV v2.2 root directory "
             "(must contain hosp/ and icu/ subdirectories).",
    )
    parser.add_argument(
        "--hirid", required=True,
        help="Path to HiRID v1.1.1 root directory "
             "(must contain general_table.csv).",
    )
    parser.add_argument(
        "--out_dir", default=str(OUT_DIR),
        help=f"Output directory for cohort CSVs (default: {OUT_DIR}).",
    )
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Build and save MIMIC-IV cohort
    mimic_cohort = build_mimic_cohort(args.mimic)
    mimic_out    = out / "mimic_cohort.csv"
    mimic_cohort.to_csv(mimic_out, index=False)
    print(f"\nMIMIC-IV cohort saved -> {mimic_out}")

    # Build and save HiRID cohort
    hirid_cohort = build_hirid_cohort(args.hirid)
    hirid_out    = out / "hirid_cohort.csv"
    hirid_cohort.to_csv(hirid_out, index=False)
    print(f"HiRID cohort saved   -> {hirid_out}")

    print("\nBoth cohorts built successfully.")
    print(f"   MIMIC-IV : {len(mimic_cohort):,} stays  "
          f"({mimic_cohort[OUTCOME].mean() * 100:.2f}% mortality)")
    print(f"   HiRID    : {len(hirid_cohort):,} stays  "
          f"({hirid_cohort[OUTCOME].mean() * 100:.2f}% mortality)")
    print(f"\nNext step:")
    print(f"  python src/idi_features.py "
          f"--cohort {mimic_out} "
          f"--events data/raw/mimic-iv/icu/chartevents.csv "
          f"--output data/processed/idi_features.csv")


if __name__ == "__main__":
    main()
