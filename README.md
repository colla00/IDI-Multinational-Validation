# IDI-Multinational-Validation

**Multinational External Validation of the Intensive Documentation Index for ICU Mortality Prediction**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![PhysioNet MIMIC-IV](https://img.shields.io/badge/Data-PhysioNet-green)](https://physionet.org/content/mimiciv/2.2/)
[![PhysioNet HiRID](https://img.shields.io/badge/Data-HiRID-green)](https://physionet.org/content/hirid/1.1.1/)

## Overview

This repository contains all analysis code for:

> **Collier AM, Shalhout SZ.** Intensive Documentation Index as an All-Cause Mortality Predictor in
> Critically Ill Patients: A Multi-Center External Validation Study.
> *npj Digital Medicine* (under review, 2026).

The Intensive Documentation Index (IDI) is a zero-burden mortality prediction framework derived
exclusively from **nursing documentation timestamps** — no laboratory values, no imaging, no
manual scoring required.

---

## Key Results

| Cohort | Dataset | N | Mortality | AUROC | AUPRC | Brier |
|--------|---------|---|-----------|-------|-------|-------|
| MIMIC-IV (HF, USA) | Derivation | 26,153 | 15.99% | 0.6401 | 0.2294 | 0.1347 |
| HiRID (All-ICU, Switzerland) | External Validation | 33,897 | 6.08% | 0.9063 | 0.4546 | 0.1168 |

- **Calibration slope (HiRID):** 0.98 (near-perfect)
- **Zero data-entry burden:** IDI uses only passively recorded timestamps
- **ΔAUROC vs baseline (MIMIC-IV):** +0.025 (p = 0.015, DeLong test)

---

## Repository Structure

```
IDI-Multinational-Validation/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── data/
│   └── README.md              # How to access MIMIC-IV and HiRID via PhysioNet
├── src/
│   ├── cohort_selection.py    # Inclusion/exclusion criteria → cohort.csv
│   ├── idi_features.py        # 9 IDI temporal features from timestamps
│   ├── leakage_filter.py      # Remove features with |r|>0.30 with ICU LOS
│   ├── model.py               # Train baseline + IDI-enhanced models
│   ├── hirid_validation.py    # External validation on HiRID cohort
│   └── metrics.py             # AUROC, DeLong, calibration slope, AUPRC, Brier
├── results/
│   ├── figures/               # ROC curves, calibration plots, feature importance
│   └── tables/                # Performance tables (CSV)
└── notebooks/
    ├── 01_cohort_selection.ipynb
    ├── 02_feature_engineering.ipynb
    ├── 03_model_training.ipynb
    ├── 04_hirid_validation.ipynb
    └── 05_figures.ipynb
```

---

## Data Access

**MIMIC-IV (v2.2):**
> https://physionet.org/content/mimiciv/2.2/
> Requires PhysioNet credentialed access + CITI training.

**HiRID (v1.1.1):**
> https://physionet.org/content/hirid/1.1.1/
> Requires PhysioNet credentialed access + data use agreement.

**No patient data is included in this repository.** Place downloaded data in `data/raw/` (excluded by `.gitignore`).

---

## Installation

```bash
git clone https://github.com/colla00/IDI-Multinational-Validation.git
cd IDI-Multinational-Validation
pip install -r requirements.txt
```

---

## Usage

```bash
# 1. Build cohorts
python src/cohort_selection.py --mimic /data/raw/mimic-iv --hirid /data/raw/hirid

# 2. Extract IDI features
python src/idi_features.py --cohort data/processed/mimic_cohort.csv

# 3. Apply leakage filter (remove |r|>0.30 with ICU LOS)
python src/leakage_filter.py

# 4. Train and evaluate models
python src/model.py --split temporal  # temporal: train 2008-2018, test 2019

# 5. HiRID external validation
python src/hirid_validation.py

# 6. Compute all metrics
python src/metrics.py
```

---

## Citation

```bibtex
@article{collier2026npj,
  title   = {Intensive Documentation Index as an All-Cause Mortality Predictor
             in Critically Ill Patients: A Multi-Center External Validation Study},
  author  = {Collier, Alexis M. and Shalhout, Sophia Z.},
  journal = {npj Digital Medicine},
  year    = {2026},
  note    = {Under review},
  doi     = {10.5281/zenodo.XXXXXXX}
}
```

---

## Competing Interests & Patents

Dr. Collier is Founder and CEO of VitaSignal LLC. This work relates to:
- USPTO Provisional App. No. **63/976,293** — NurseRhythm IDI Engine (filed Feb 2026)
- USPTO Provisional App. No. **63/946,187** — CDS-EQUITY (filed Dec 2025)
- USPTO Provisional App. No. **63/932,953** — CRIS-E (filed Dec 2025)

Licensing: info@vitasignal.ai | NIH Award No. 1OT2OD032581 (Bayh-Dole applies)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
