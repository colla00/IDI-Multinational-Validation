# Multinational External Validation of the Intensive Documentation Index for ICU Mortality Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/colla00/IDI-Multinational-Validation/blob/main/LICENSE)
[![PhysioNet MIMIC-IV](https://img.shields.io/badge/Data-MIMIC--IV%20v2.2-blue)](https://physionet.org/content/mimiciv/2.2/)
[![PhysioNet HiRID](https://img.shields.io/badge/Data-HiRID%20v1.1.1-blue)](https://physionet.org/content/hirid/1.1.1/)

## Overview

This repository contains all analysis code for:

> **Collier AM, Shalhout SZ.** Intensive Documentation Index as an All-Cause Mortality Predictor in Critically Ill Patients: A Multi-Center External Validation Study. *Journal of Biomedical Informatics* (under review, 2026).

The Intensive Documentation Index (IDI) is a zero-burden mortality prediction framework derived exclusively from **nursing documentation timestamps** — no laboratory values, no imaging, no manual scoring required.

---

## Key Results

| Cohort | Dataset | N (model) | Mortality | AUROC | AUPRC | Brier |
|--------|---------|-----------|-----------|-------|-------|-------|
| MIMIC-IV (HF, USA) | Derivation | 26,133 complete-case (26,153 eligible) | 15.99% | 0.6491 | 0.2530 | 0.1299 |
| HiRID (All-ICU, Switzerland) | External Validation | 33,897 | 6.08% | 0.9063 | 0.4546 | 0.1168 |

- **Calibration slope (MIMIC-IV):** Baseline 1.07, IDI-Enhanced 1.05
- **Calibration slope (HiRID):** 0.98 (near-perfect)
- **Zero data-entry burden:** IDI uses only passively recorded timestamps
- **AUROC vs baseline (MIMIC-IV):** +0.0249 (p < 0.001, DeLong test)
- **Split:** 80/20 stratified random split (seed=42); train n=20,906, test n=5,227
- **Leakage-corrected:** idi_events_24h and idi_events_per_hour excluded (|r|>0.30 with ICU LOS)

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

# 4. Train and evaluate models (80/20 stratified random split, seed=42)
python src/model.py --split random --seed 42

# 5. HiRID external validation
python src/hirid_validation.py

# 6. Compute all metrics
python src/metrics.py
```

---

## Citation

```bibtex
@article{collier2026jbi,
  title   = {Intensive Documentation Index as an All-Cause Mortality Predictor
             in Critically Ill Patients: A Multi-Center External Validation Study},
  author  = {Collier, Alexis M. and Shalhout, Sophia Z.},
  journal = {Journal of Biomedical Informatics},
  year    = {2026},
  note    = {Under review}
}
```

---

## Competing Interests & Patents

Dr. Collier is Founder and CEO of VitaSignal LLC. Multiple U.S. provisional patent applications related to this work are pending. Licensing inquiries: info@vitasignal.ai

---

## Funding

This research was, in part, funded by the National Institutes of Health through the NIH AIM-AHEAD program.

---

## License

MIT License — see [LICENSE](https://github.com/colla00/IDI-Multinational-Validation/blob/main/LICENSE) for details.
