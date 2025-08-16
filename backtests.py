from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import chi2

def hits(realized: pd.Series, var_series: pd.Series) -> np.ndarray:
    return (realized < -var_series).astype(int).values

def kupiec_test(hits_arr: np.ndarray, alpha: float) -> float:
    T = len(hits_arr)
    N = hits_arr.sum()
    p_hat = N / T if 0 < N < T else (1e-6 if N == 0 else 1 - 1e-6)
    L0 = ((1 - alpha) ** (T - N)) * (alpha ** N)
    L1 = ((1 - p_hat) ** (T - N)) * (p_hat ** N)
    LR_uc = -2 * np.log(L0 / L1)
    return 1 - chi2.cdf(LR_uc, df=1)

def christoffersen_test(hits_arr: np.ndarray, alpha: float) -> float:
    T = len(hits_arr)
    N00 = N01 = N10 = N11 = 0
    for t in range(1, T):
        prev, curr = hits_arr[t-1], hits_arr[t]
        if prev == 0 and curr == 0: N00 += 1
        if prev == 0 and curr == 1: N01 += 1
        if prev == 1 and curr == 0: N10 += 1
        if prev == 1 and curr == 1: N11 += 1
    pi_hat = hits_arr.mean()
    pi0_hat = N01 / (N00 + N01) if (N00 + N01) > 0 else 0.0
    pi1_hat = N11 / (N10 + N11) if (N10 + N11) > 0 else 0.0

    # Log-likelihoods
    eps = 1e-12
    li = 0.0
    li += (N00 + N01) * np.log(max(1 - pi_hat, eps))
    li += (N10 + N11) * np.log(max(pi_hat, eps))
    li += (N01 + N11) * np.log(max(pi_hat, eps))
    li += (N00 + N10) * np.log(max(1 - pi_hat, eps))

    lc = 0.0
    lc += N00 * np.log(max(1 - pi0_hat, eps))
    lc += N01 * np.log(max(pi0_hat, eps))
    lc += N10 * np.log(max(1 - pi1_hat, eps))
    lc += N11 * np.log(max(pi1_hat, eps))

    LR_ind = -2 * (li - lc)

    # Kupiec
    N = hits_arr.sum()
    p_hat = N / T if (0 < N < T) else (1e-6 if N == 0 else 1 - 1e-6)
    L0 = ((1 - alpha) ** (T - N)) * (alpha ** N)
    L1 = ((1 - p_hat) ** (T - N)) * (p_hat ** N)
    LR_uc = -2 * np.log(L0 / L1)

    LR_cc = LR_uc + LR_ind
    return 1 - chi2.cdf(LR_cc, df=2)

def summarize_backtests(realized: pd.Series, var_dict: dict, alpha: float) -> pd.DataFrame:
    out = []
    for name, v in var_dict.items():
        h = hits(realized, v)
        out.append({
            "Method": name,
            "Observed_Hits": int(h.sum()),
            "Empirical_Rate": float(h.mean()),
            "Kupiec_p": float(kupiec_test(h, alpha)),
            "Christoffersen_p": float(christoffersen_test(h, alpha)),
        })
    return pd.DataFrame(out).sort_values("Method").reset_index(drop=True)
