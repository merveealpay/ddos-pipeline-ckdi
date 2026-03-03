"""
ckdi.py – Composite Knowledge Drift Index
==========================================

CKDI measures how much a given traffic class has *drifted* away from the
normal (benign) baseline.  It combines two complementary views:

1. **Statistical drift (Δ_stat)**
   Kolmogorov–Smirnov (KS) statistic averaged across all features.
   KS ∈ [0, 1]; 0 = identical distributions, 1 = fully separated.

2. **PCA-space drift (Δ_pca)**
   Both the baseline and the attack class are projected onto the PCA
   components learned from the baseline.  Drift is the normalised
   Euclidean distance between the centroid of the attack cloud and the
   centroid of the baseline cloud in PCA space.

Final CKDI score::

    CKDI = α · Δ_stat + (1 - α) · Δ_pca        α ∈ [0, 1]  (default 0.5)

The score is clipped to [0, 1].

Usage
-----
>>> from src.ckdi import compute_ckdi
>>> results = compute_ckdi(baseline_df, attacks_dict, feature_cols, alpha=0.5)
>>> # results is a dict  { attack_label: ckdi_score }
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Number of PCA components retained for drift measurement
_N_COMPONENTS = 10


def compute_ckdi(
    baseline: pd.DataFrame,
    attacks: dict[str, pd.DataFrame],
    feature_cols: list[str],
    alpha: float = 0.5,
    n_components: Optional[int] = None,
) -> dict[str, float]:
    """Compute CKDI for every attack class relative to the benign baseline.

    Parameters
    ----------
    baseline:
        DataFrame of benign/normal traffic (feature columns + Label).
    attacks:
        Dict mapping attack-class name → DataFrame of that class.
    feature_cols:
        Names of the numeric feature columns to use.
    alpha:
        Weight of the statistical drift component (1-alpha goes to PCA drift).
        Must be in [0, 1].
    n_components:
        PCA components to retain.  Defaults to min(10, n_features, n_samples-1).

    Returns
    -------
    dict[str, float]
        ``{attack_label: ckdi_score}`` where score ∈ [0, 1].
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    X_base = baseline[feature_cols].values.astype(float)

    # ---------- Fit scaler & PCA on baseline ---------------------------------
    scaler = StandardScaler()
    X_base_scaled = scaler.fit_transform(X_base)

    n_comp = n_components or min(_N_COMPONENTS, X_base.shape[1], X_base.shape[0] - 1)
    pca = PCA(n_components=n_comp)
    X_base_pca = pca.fit_transform(X_base_scaled)
    base_centroid_pca = X_base_pca.mean(axis=0)

    # Reference spread: std of baseline projections (for normalisation)
    base_spread = X_base_pca.std(axis=0).mean() + 1e-9

    results: dict[str, float] = {}

    for label, df_atk in attacks.items():
        X_atk = df_atk[feature_cols].values.astype(float)
        X_atk_scaled = scaler.transform(X_atk)

        # -- Statistical drift (KS) ------------------------------------------
        ks_scores: list[float] = []
        for j in range(X_base.shape[1]):
            stat, _ = ks_2samp(X_base[:, j], X_atk[:, j])
            ks_scores.append(stat)
        delta_stat = float(np.mean(ks_scores))

        # -- PCA-space drift --------------------------------------------------
        X_atk_pca = pca.transform(X_atk_scaled)
        atk_centroid_pca = X_atk_pca.mean(axis=0)
        pca_dist = float(np.linalg.norm(atk_centroid_pca - base_centroid_pca))
        # Normalise by baseline spread so the metric is unitless & comparable
        delta_pca = float(np.tanh(pca_dist / base_spread))  # squashes to (0,1)

        # -- Composite score --------------------------------------------------
        ckdi = float(np.clip(alpha * delta_stat + (1.0 - alpha) * delta_pca, 0.0, 1.0))
        results[label] = ckdi

        logger.debug(
            "CKDI[%s]: Δ_stat=%.4f  Δ_pca=%.4f  CKDI=%.4f",
            label, delta_stat, delta_pca, ckdi,
        )

    return results


def compute_ckdi_detailed(
    baseline: pd.DataFrame,
    attacks: dict[str, pd.DataFrame],
    feature_cols: list[str],
    alpha: float = 0.5,
    n_components: Optional[int] = None,
) -> pd.DataFrame:
    """Return a detailed DataFrame with sub-components alongside CKDI.

    Columns: ``attack_class``, ``delta_stat``, ``delta_pca``, ``ckdi``.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    X_base = baseline[feature_cols].values.astype(float)

    scaler = StandardScaler()
    X_base_scaled = scaler.fit_transform(X_base)

    n_comp = n_components or min(_N_COMPONENTS, X_base.shape[1], X_base.shape[0] - 1)
    pca = PCA(n_components=n_comp)
    X_base_pca = pca.fit_transform(X_base_scaled)
    base_centroid_pca = X_base_pca.mean(axis=0)
    base_spread = X_base_pca.std(axis=0).mean() + 1e-9

    # Also capture explained variance
    explained_var = float(pca.explained_variance_ratio_.sum())

    rows: list[dict] = []
    for label, df_atk in attacks.items():
        X_atk = df_atk[feature_cols].values.astype(float)
        X_atk_scaled = scaler.transform(X_atk)

        ks_scores: list[float] = []
        for j in range(X_base.shape[1]):
            stat, _ = ks_2samp(X_base[:, j], X_atk[:, j])
            ks_scores.append(stat)
        delta_stat = float(np.mean(ks_scores))

        X_atk_pca = pca.transform(X_atk_scaled)
        atk_centroid_pca = X_atk_pca.mean(axis=0)
        pca_dist = float(np.linalg.norm(atk_centroid_pca - base_centroid_pca))
        delta_pca = float(np.tanh(pca_dist / base_spread))

        ckdi = float(np.clip(alpha * delta_stat + (1.0 - alpha) * delta_pca, 0.0, 1.0))

        rows.append({
            "attack_class": label,
            "delta_stat": round(delta_stat, 6),
            "delta_pca": round(delta_pca, 6),
            "pca_explained_var": round(explained_var, 6),
            "alpha": alpha,
            "ckdi": round(ckdi, 6),
        })

    return pd.DataFrame(rows).sort_values("ckdi", ascending=False).reset_index(drop=True)
