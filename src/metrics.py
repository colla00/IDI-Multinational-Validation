"""
metrics.py
----------
Performance metrics for the multinational IDI validation study.

Functions:
  delong_test()         - DeLong test for AUROC comparison (Hanley-McNeil variance)
  calibration_slope()   - logistic calibration slope and intercept
  bootstrap_auroc_ci()  - bootstrap 95% CI for AUROC
  full_metrics()        - compute and print the full performance table

Note on DeLong test: delong_test() uses the Hanley-McNeil variance formula
(var1 + var2), which assumes the two AUROCs are independent. For correlated
models evaluated on the same test set, the correct approach includes a
covariance term (DeLong et al., Biometrics 1988). The independence approximation
is slightly conservative, producing wider confidence intervals and larger
p-values. This is the standard approach in clinical ML literature and is the
safer direction for reported p-values.

Reference:
  Collier AM, Shalhout SZ. Intensive Documentation Index as an All-Cause
  Mortality Predictor in Critically Ill Patients: A Multi-Center External
  Validation Study. Journal of Biomedical Informatics (under review, 2026).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                              average_precision_score)
from sklearn.linear_model import LogisticRegression
from scipy import stats


def delong_test(y_true, y_pred1, y_pred2):
    """
    DeLong test for comparing two AUROCs (Hanley-McNeil variance formula).
    Returns (z_stat, p_value, auc1, auc2, delta_auc).

    Uses var1 + var2 (independence assumption) -- slightly conservative for
    correlated AUROCs on the same test set. See module docstring.
    """
    y_true  = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1)
    y_pred2 = np.asarray(y_pred2)

    auc1 = roc_auc_score(y_true, y_pred1)
    auc2 = roc_auc_score(y_true, y_pred2)

    n1 = int(y_true.sum())
    n0 = int((1 - y_true).sum())

    def _hanley_mcneil_var(auc, n1, n0):
        q1 = auc / (2 - auc)
        q2 = 2 * auc ** 2 / (1 + auc)
        return (auc * (1 - auc)
                + (n1 - 1) * (q1 - auc ** 2)
                + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0)

    var1 = _hanley_mcneil_var(auc1, n1, n0)
    var2 = _hanley_mcneil_var(auc2, n1, n0)

    se = np.sqrt(var1 + var2)
    z  = (auc2 - auc1) / se if se > 0 else 0.0
    p  = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p, auc1, auc2, auc2 - auc1


def calibration_slope(y_true, y_pred):
    """
    Logistic regression of outcome on logit(predicted probability).
    Returns (slope, intercept). Slope = 1.0 indicates perfect calibration.
    """
    eps      = 1e-6
    log_odds = np.log(
        np.clip(y_pred, eps, 1 - eps) /
        (1 - np.clip(y_pred, eps, 1 - eps))
    )
    lr = LogisticRegression(fit_intercept=True, max_iter=1000)
    lr.fit(log_odds.reshape(-1, 1), np.asarray(y_true))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def bootstrap_auroc_ci(y_true, y_pred, n_boot=1000, ci=0.95):
    """
    Bootstrap percentile 95% CI for AUROC.
    Returns (lower, upper) as floats.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    aucs   = []
    n      = len(y_true)
    rng    = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    alpha  = (1 - ci) / 2
    lo, hi = np.percentile(aucs, [alpha * 100, (1 - alpha) * 100])
    return float(lo), float(hi)


def full_metrics(y_true, y_pred_baseline, y_pred_idi, label=""):
    """
    Compute and print the full performance metric table.
    Returns a dict with all key statistics.
    """
    y_true          = np.asarray(y_true)
    y_pred_baseline = np.asarray(y_pred_baseline)
    y_pred_idi      = np.asarray(y_pred_idi)

    auc_b = roc_auc_score(y_true, y_pred_baseline)
    auc_i = roc_auc_score(y_true, y_pred_idi)
    ci_b  = bootstrap_auroc_ci(y_true, y_pred_baseline)
    ci_i  = bootstrap_auroc_ci(y_true, y_pred_idi)
    _, p, _, _, delta = delong_test(y_true, y_pred_baseline, y_pred_idi)

    brier_b = brier_score_loss(y_true, y_pred_baseline)
    brier_i = brier_score_loss(y_true, y_pred_idi)
    auprc_i = average_precision_score(y_true, y_pred_idi)

    cal_b_slope, cal_b_int = calibration_slope(y_true, y_pred_baseline)
    cal_i_slope, cal_i_int = calibration_slope(y_true, y_pred_idi)

    print(f"\n{'=' * 55}")
    if label:
        print(f"  {label}")
    print(f"  Baseline AUROC:    {auc_b:.4f}  "
          f"(95% CI {ci_b[0]:.3f}-{ci_b[1]:.3f})")
    print(f"  IDI AUROC:         {auc_i:.4f}  "
          f"(95% CI {ci_i[0]:.3f}-{ci_i[1]:.3f})")
    print(f"  ΔAUROC:            {delta:+.4f}  (p={p:.4f}, DeLong)")
    print(f"  AUPRC (IDI):       {auprc_i:.4f}")
    print(f"  Brier (baseline):  {brier_b:.4f}")
    print(f"  Brier (IDI):       {brier_i:.4f}")
    print(f"  Cal slope (base):  {cal_b_slope:.3f}  "
          f"(intercept {cal_b_int:.3f})")
    print(f"  Cal slope (IDI):   {cal_i_slope:.3f}  "
          f"(intercept {cal_i_int:.3f})")
    print(f"{'=' * 55}")

    return dict(
        auc_b=auc_b,             auc_i=auc_i,
        ci_b=ci_b,               ci_i=ci_i,
        delta=delta,             p=p,
        auprc=auprc_i,
        brier_b=brier_b,         brier_i=brier_i,
        cal_b_slope=cal_b_slope, cal_b_int=cal_b_int,
        cal_i_slope=cal_i_slope, cal_i_int=cal_i_int,
    )
