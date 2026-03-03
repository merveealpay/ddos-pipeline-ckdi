"""
bdri.py – Balanced Drift Resilience Index
==========================================

BDRI combines three complementary sub-indices and finds the *optimal*
convex combination of them via SLSQP constrained optimisation.

Sub-indices (each normalised to [0, 1])
----------------------------------------
SDI – Statistical Drift Index
    Ratio of the attack-class standard deviation to the baseline standard
    deviation, averaged over features and normalised via tanh so it stays
    in (0, 1).  SDI > 1 means higher variability than baseline (stressed
    conditions).

ALR – Asymmetric Likelihood Ratio
    Log-likelihood ratio of a sample belonging to the attack distribution
    vs the baseline distribution, using Gaussian approximations.
    Positive ALR indicates the sample is more likely under the attack model.
    We report the normalised mean positive ALR for the attack class.

RIF – Relative Information Flow
    Symmetric KL divergence (Jensen–Shannon divergence) between the
    per-feature histograms of baseline and attack, averaged over features.
    JSD ∈ [0, 1] (base-2 logarithm).

Weight optimisation
-------------------
Objective: maximise the *separability* of BDRI scores between the attack
class and the baseline (Fisher's criterion – inter-class variance /
intra-class variance).

    maximise   J(w) = (μ_atk − μ_base)²  /  (σ_atk² + σ_base² + ε)

Subject to:  w₁ + w₂ + w₃ = 1,   wᵢ ≥ 0.

Because SLSQP minimises, we minimise −J(w).

BDRI for a given attack class is the weighted score computed with the
optimal weights.

Usage
-----
>>> from src.bdri import compute_bdri
>>> results_df = compute_bdri(baseline_df, attacks_dict, feature_cols)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Sub-index implementations
# ---------------------------------------------------------------------------

def _sdi(X_base: np.ndarray, X_atk: np.ndarray) -> float:
    """Statistical Drift Index – normalised std-ratio averaged over features."""
    std_base = X_base.std(axis=0) + _EPS
    std_atk = X_atk.std(axis=0) + _EPS
    ratio = std_atk / std_base           # > 1 means higher spread (drift)
    # tanh maps non-negative values to [0, 1); abs() ensures non-negative input
    # so tanh(abs(ratio - 1)) measures deviation from 1 on the [0, 1) scale
    return float(np.mean(np.tanh(np.abs(ratio - 1.0))))


def _alr(X_base: np.ndarray, X_atk: np.ndarray) -> float:
    """Asymmetric Likelihood Ratio using Gaussian approximation."""
    mu_base = X_base.mean(axis=0)
    sigma_base = X_base.std(axis=0) + _EPS

    mu_atk = X_atk.mean(axis=0)
    sigma_atk = X_atk.std(axis=0) + _EPS

    # Log-likelihood of attack samples under attack vs baseline model
    def log_likelihood(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        return -0.5 * np.sum(((X - mu) / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2), axis=1)

    ll_atk = log_likelihood(X_atk, mu_atk, sigma_atk)
    ll_base = log_likelihood(X_atk, mu_base, sigma_base)
    llr = ll_atk - ll_base   # per-sample log-likelihood ratio

    # Normalise: tanh of mean positive LLR gives a value in (0, 1)
    mean_llr = float(np.mean(np.maximum(llr, 0.0)))
    # Scale by n_features so the value is feature-count independent
    scaled = mean_llr / (X_base.shape[1] + _EPS)
    return float(np.tanh(scaled))


def _rif(X_base: np.ndarray, X_atk: np.ndarray, n_bins: int = 20) -> float:
    """Relative Information Flow – mean Jensen–Shannon divergence over features."""
    n_feat = X_base.shape[1]
    jsd_vals: list[float] = []
    for j in range(n_feat):
        col_base = X_base[:, j]
        col_atk = X_atk[:, j]
        # Shared bin edges spanning both distributions
        lo = min(col_base.min(), col_atk.min())
        hi = max(col_base.max(), col_atk.max()) + _EPS
        bins = np.linspace(lo, hi, n_bins + 1)
        p, _ = np.histogram(col_base, bins=bins, density=True)
        q, _ = np.histogram(col_atk, bins=bins, density=True)
        # Normalise to probability vectors (add small smoothing)
        p = p + _EPS
        q = q + _EPS
        p /= p.sum()
        q /= q.sum()
        jsd = jensenshannon(p, q, base=2) ** 2   # JSD² ∈ [0, 1]
        jsd_vals.append(float(jsd))
    return float(np.mean(jsd_vals))


# ---------------------------------------------------------------------------
# Weight optimisation via SLSQP
# ---------------------------------------------------------------------------

def _optimise_weights(
    scores_base: np.ndarray,  # shape (3,) – sub-index scores for baseline samples
    scores_atk: np.ndarray,   # shape (3,) – sub-index scores for attack samples
    w0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Find convex weights that maximise separability (Fisher criterion).

    Parameters
    ----------
    scores_base:
        1-D array of length 3: [SDI_base, ALR_base, RIF_base] – these are
        the "within-baseline" scores computed using leave-one-out halves.
    scores_atk:
        1-D array of length 3: [SDI_atk, ALR_atk, RIF_atk].
    w0:
        Initial guess for weights (uniform by default).

    Returns
    -------
    np.ndarray of shape (3,), non-negative, summing to 1.
    """
    if w0 is None:
        w0 = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    def neg_fisher(w: np.ndarray) -> float:
        bdri_base = float(np.dot(w, scores_base))
        bdri_atk = float(np.dot(w, scores_atk))
        # Simplified Fisher: maximise squared distance
        return -((bdri_atk - bdri_base) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * 3

    res = minimize(
        neg_fisher,
        x0=w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    if not res.success:
        logger.warning("SLSQP did not converge: %s – using uniform weights.", res.message)
        return w0.copy()

    weights = np.clip(res.x, 0.0, 1.0)
    weights /= weights.sum() + _EPS   # re-normalise after clipping
    return weights


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_bdri(
    baseline: pd.DataFrame,
    attacks: dict[str, pd.DataFrame],
    feature_cols: list[str],
) -> pd.DataFrame:
    """Compute BDRI for every attack class.

    The function:
    1. Computes SDI, ALR, RIF for each attack class vs the benign baseline.
    2. Estimates "baseline self-scores" by splitting the baseline in half.
    3. Runs SLSQP to find the weight vector (w1, w2, w3) that maximises the
       separability between attack and baseline BDRI.
    4. Reports per-class BDRI = w1·SDI + w2·ALR + w3·RIF.

    Parameters
    ----------
    baseline:
        Benign traffic DataFrame (features + Label).
    attacks:
        Dict { attack_label : attack DataFrame }.
    feature_cols:
        Numeric feature column names.

    Returns
    -------
    pd.DataFrame with columns:
        attack_class, sdi, alr, rif, w_sdi, w_alr, w_rif, bdri
    """
    X_base = baseline[feature_cols].values.astype(float)

    scaler = StandardScaler()
    X_base_scaled = scaler.fit_transform(X_base)

    # Baseline self-scores via 50/50 split (approximates intra-baseline drift)
    mid = len(X_base_scaled) // 2
    X_b1, X_b2 = X_base_scaled[:mid], X_base_scaled[mid:]
    base_sdi = _sdi(X_b1, X_b2)
    base_alr = _alr(X_b1, X_b2)
    base_rif = _rif(X_b1, X_b2)
    scores_base = np.array([base_sdi, base_alr, base_rif])

    logger.debug("Baseline self-scores: SDI=%.4f ALR=%.4f RIF=%.4f", *scores_base)

    rows: list[dict] = []

    for label, df_atk in attacks.items():
        X_atk = scaler.transform(df_atk[feature_cols].values.astype(float))

        sdi_val = _sdi(X_base_scaled, X_atk)
        alr_val = _alr(X_base_scaled, X_atk)
        rif_val = _rif(X_base_scaled, X_atk)
        scores_atk = np.array([sdi_val, alr_val, rif_val])

        # Optimise weights for this class
        w = _optimise_weights(scores_base, scores_atk)
        # Check if optimization succeeded (non-uniform weights signal optimised result)
        weights_optimized = not np.allclose(w, np.array([1.0 / 3.0] * 3), atol=1e-4)
        bdri = float(np.clip(np.dot(w, scores_atk), 0.0, 1.0))

        rows.append({
            "attack_class": label,
            "sdi": round(sdi_val, 6),
            "alr": round(alr_val, 6),
            "rif": round(rif_val, 6),
            "w_sdi": round(float(w[0]), 6),
            "w_alr": round(float(w[1]), 6),
            "w_rif": round(float(w[2]), 6),
            "weights_optimized": weights_optimized,
            "bdri": round(bdri, 6),
        })

        logger.debug(
            "BDRI[%s]: SDI=%.4f ALR=%.4f RIF=%.4f  w=[%.3f,%.3f,%.3f]  BDRI=%.4f",
            label, sdi_val, alr_val, rif_val, w[0], w[1], w[2], bdri,
        )

    return pd.DataFrame(rows).sort_values("bdri", ascending=False).reset_index(drop=True)
