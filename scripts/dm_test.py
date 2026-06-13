#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import scipy.stats

def dm_test(actual_lst, pred1_lst, pred2_lst, h=1, power=1):
    """
    Computes the Diebold-Mariano test statistic for two sets of predictions.
    Null hypothesis: No difference in the accuracy of the two competing forecasts.
    """
    actual = np.array(actual_lst)
    pred1 = np.array(pred1_lst)
    pred2 = np.array(pred2_lst)
    
    e1 = actual - pred1
    e2 = actual - pred2
    
    d = (np.abs(e1))**power - (np.abs(e2))**power
    
    mean_d = np.mean(d)
    
    def autocovariance(xi, k):
        n = len(xi)
        if k >= n:
            return 0
        x_mean = np.mean(xi)
        return np.sum((xi[:n-k] - x_mean) * (xi[k:] - x_mean)) / n
        
    gamma = [autocovariance(d, k) for k in range(h)]
    var_d = gamma[0] + 2 * sum(gamma[1:])
    
    if var_d == 0:
        return 0, 1.0
        
    n = len(d)
    DM_stat = mean_d / np.sqrt(var_d / n)
    p_value = 2 * scipy.stats.t.cdf(-np.abs(DM_stat), df=n-1)
    
    return DM_stat, p_value

if __name__ == "__main__":
    try:
        df = pd.read_csv("results_detailed.csv")
    except FileNotFoundError:
        print("Please run scripts/run_definitive_experiment.py first.")
        sys.exit(1)
        
    actual = df["market_price"].values
    bs_preds = df["bs_price"].values
    crr_preds = df["crr_price"].values
    mc_preds = df["mc_price"].values
    
    print("=" * 60)
    print("  DIEBOLD-MARIANO TEST — CLASSICAL METHODS")
    print("=" * 60)
    
    # BS vs CRR
    dm_stat, p_val = dm_test(actual, bs_preds, crr_preds)
    print(f"  BS vs CRR : DM-stat = {dm_stat:>8.4f} | p-value = {p_val:>8.4f}")
    if p_val > 0.05:
        print("    -> Fail to reject null (Methods are statistically tied)")
    else:
        print("    -> Reject null (Difference is significant)")
        
    # BS vs MC
    dm_stat, p_val = dm_test(actual, bs_preds, mc_preds)
    print(f"\n  BS vs MC  : DM-stat = {dm_stat:>8.4f} | p-value = {p_val:>8.4f}")
    if p_val > 0.05:
        print("    -> Fail to reject null (Methods are statistically tied)")
    else:
        print("    -> Reject null (Difference is significant)")

    # CRR vs MC
    dm_stat, p_val = dm_test(actual, crr_preds, mc_preds)
    print(f"\n  CRR vs MC : DM-stat = {dm_stat:>8.4f} | p-value = {p_val:>8.4f}")
    if p_val > 0.05:
        print("    -> Fail to reject null (Methods are statistically tied)")
    else:
        print("    -> Reject null (Difference is significant)")
        
    print("=" * 60)
