"""
Performance Metrics — npj Multinational Validation
AUROC, DeLong test, calibration slope, AUPRC, Brier score.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                              average_precision_score)
from scipy import stats


def delong_test(y_true, y_pred1, y_pred2):
    """
    DeLong's test for comparing two correlated AUROCs.
    Returns (z_stat, p_value, auc1, auc2, delta_auc).
    """
    auc1 = roc_auc_score(y_true, y_pred1)
    auc2 = roc_auc_score(y_true, y_pred2)

    n1 = int(y_true.sum())     # positives
    n0 = int((1-y_true).sum()) # negatives

    # Wilcoxon-Mann-Whitney statistic variance (simplified)
    q1 = auc1 / (2 - auc1)
    q2 = 2 * auc1**2 / (1 + auc1)
    var1 = (auc1*(1-auc1) + (n1-1)*(q1-auc1**2) + (n0-1)*(q2-auc1**2)) / (n1*n0)

    q1b = auc2 / (2 - auc2)
    q2b = 2 * auc2**2 / (1 + auc2)
    var2 = (auc2*(1-auc2) + (n1-1)*(q1b-auc2**2) + (n0-1)*(q2b-auc2**2)) / (n1*n0)

    se = np.sqrt(var1 + var2)
    z  = (auc2 - auc1) / se if se > 0 else 0.0
    p  = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p, auc1, auc2, auc2 - auc1


def calibration_slope(y_true, y_pred):
    from sklearn.linear_model import LogisticRegression
    log_odds = np.log(np.clip(y_pred, 1e-6, 1-1e-6) /
                      (1 - np.clip(y_pred, 1e-6, 1-1e-6)))
    lr = LogisticRegression(fit_intercept=True)
    lr.fit(log_odds.reshape(-1, 1), y_true)
    return float(lr.coef_[0][0])


def bootstrap_auroc_ci(y_true, y_pred, n_boot=1000, ci=0.95):
    aucs = []
    n = len(y_true)
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    alpha = (1 - ci) / 2
    return np.percentile(aucs, [alpha*100, (1-alpha)*100])


def full_metrics(y_true, y_pred_baseline, y_pred_idi, label=""):
    auc_b  = roc_auc_score(y_true, y_pred_baseline)
    auc_i  = roc_auc_score(y_true, y_pred_idi)
    ci_b   = bootstrap_auroc_ci(y_true, y_pred_baseline)
    ci_i   = bootstrap_auroc_ci(y_true, y_pred_idi)
    _, p, _, _, delta = delong_test(y_true, y_pred_baseline, y_pred_idi)
    brier_b = brier_score_loss(y_true, y_pred_baseline)
    brier_i = brier_score_loss(y_true, y_pred_idi)
    auprc_i = average_precision_score(y_true, y_pred_idi)
    cal_b   = calibration_slope(y_true, y_pred_baseline)
    cal_i   = calibration_slope(y_true, y_pred_idi)

    print(f"\n{'='*50}")
    if label: print(f"  {label}")
    print(f"  Baseline AUROC:   {auc_b:.4f} (95% CI {ci_b[0]:.2f}-{ci_b[1]:.2f})")
    print(f"  IDI AUROC:        {auc_i:.4f} (95% CI {ci_i[0]:.2f}-{ci_i[1]:.2f})")
    print(f"  ΔAUROC:           +{delta:.4f} (p={p:.4f}, DeLong)")
    print(f"  AUPRC (IDI):      {auprc_i:.4f}")
    print(f"  Brier (baseline): {brier_b:.4f}")
    print(f"  Brier (IDI):      {brier_i:.4f}")
    print(f"  Cal slope (base): {cal_b:.3f}")
    print(f"  Cal slope (IDI):  {cal_i:.3f}")
    return dict(auc_b=auc_b, auc_i=auc_i, delta=delta, p=p,
                auprc=auprc_i, brier_b=brier_b, brier_i=brier_i,
                cal_b=cal_b, cal_i=cal_i)
