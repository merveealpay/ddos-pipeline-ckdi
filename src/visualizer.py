"""
visualizer.py – Publication-ready figures for the CKDI/BDRI pipeline
=====================================================================

Produces five PDF figures suitable for direct inclusion in an Overleaf
LaTeX document:

Figure 1 – CKDI bar chart (one bar per attack class, sub-components stacked)
Figure 2 – BDRI bar chart with optimised weight annotations
Figure 3 – CKDI vs BDRI scatter plot (one point per attack class)
Figure 4 – PCA 2-D projection: baseline + all attack classes coloured
Figure 5 – Weight distribution heatmap (w_SDI / w_ALR / w_RIF per class)
"""

from __future__ import annotations

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe in all environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared aesthetics
# ---------------------------------------------------------------------------

_PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#264653", "#A8DADC", "#6D6875",
]
# Visual parameters for PCA projection
_BENIGN_ALPHA = 0.6
_ATTACK_ALPHA = 0.5
_ATTACK_MARKERS = ["s", "^", "D", "v", "P", "X", "h", "*", "<", ">"]
_FIGSIZE = (7, 4.5)   # width × height in inches – fits a LaTeX column nicely
_DPI = 150

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": _DPI,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    logger.info("Saved figure → %s", path)


# ---------------------------------------------------------------------------
# Figure 1 – CKDI stacked bar chart
# ---------------------------------------------------------------------------

def plot_ckdi_bars(ckdi_df: pd.DataFrame, out_path: str) -> None:
    """Stacked bar: Δ_stat (blue) + Δ_pca (orange) = CKDI, per attack class.

    Parameters
    ----------
    ckdi_df : pd.DataFrame
        Output of ``compute_ckdi_detailed`` – columns include
        ``attack_class``, ``delta_stat``, ``delta_pca``, ``ckdi``.
    out_path : str
        Destination PDF path.
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    labels = ckdi_df["attack_class"].tolist()
    x = np.arange(len(labels))
    w = 0.55

    alpha_val = float(ckdi_df["alpha"].iloc[0]) if "alpha" in ckdi_df.columns else 0.5

    stat_vals = (ckdi_df["delta_stat"] * alpha_val).values
    pca_vals = (ckdi_df["delta_pca"] * (1 - alpha_val)).values

    ax.bar(x, stat_vals, width=w, label=r"$\alpha \cdot \Delta_{\mathrm{stat}}$",
           color="#457B9D", zorder=3)
    ax.bar(x, pca_vals, width=w, bottom=stat_vals,
           label=r"$(1-\alpha) \cdot \Delta_{\mathrm{pca}}$",
           color="#E9C46A", zorder=3)

    for i, row in ckdi_df.iterrows():
        ax.text(i, float(row["ckdi"]) + 0.01, f'{row["ckdi"]:.3f}',
                ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([_wrap(l) for l in labels], rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.set_ylabel("CKDI")
    ax.set_title("Composite Knowledge Drift Index (CKDI) per Attack Class")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    fig.tight_layout()
    _savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 2 – BDRI bar chart with weight annotations
# ---------------------------------------------------------------------------

def plot_bdri_bars(bdri_df: pd.DataFrame, out_path: str) -> None:
    """Bar chart of BDRI scores with per-bar weight annotation."""
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    labels = bdri_df["attack_class"].tolist()
    x = np.arange(len(labels))
    w = 0.55

    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(labels))]
    bars = ax.bar(x, bdri_df["bdri"].values, width=w, color=colors, zorder=3)

    for bar, (_, row) in zip(bars, bdri_df.iterrows()):
        ht = bar.get_height()
        annotation = (
            f"BDRI={row['bdri']:.3f}\n"
            f"w=({row['w_sdi']:.2f},{row['w_alr']:.2f},{row['w_rif']:.2f})"
        )
        ax.text(bar.get_x() + bar.get_width() / 2, ht + 0.01,
                annotation, ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels([_wrap(l) for l in labels], rotation=30, ha="right")
    ax.set_ylim(0, 1.25)
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.set_ylabel("BDRI")
    ax.set_title("Balanced Drift Resilience Index (BDRI) per Attack Class")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    fig.tight_layout()
    _savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 3 – CKDI vs BDRI scatter
# ---------------------------------------------------------------------------

def plot_ckdi_vs_bdri(
    ckdi_df: pd.DataFrame,
    bdri_df: pd.DataFrame,
    out_path: str,
) -> None:
    """Scatter plot: CKDI (x-axis) vs BDRI (y-axis), one point per class."""
    merged = ckdi_df[["attack_class", "ckdi"]].merge(
        bdri_df[["attack_class", "bdri"]], on="attack_class"
    )
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for i, row in merged.iterrows():
        c = _PALETTE[i % len(_PALETTE)]
        ax.scatter(row["ckdi"], row["bdri"], color=c, s=80, zorder=5)
        ax.annotate(
            _wrap(row["attack_class"], max_len=18),
            (row["ckdi"], row["bdri"]),
            textcoords="offset points", xytext=(6, 3),
            fontsize=7,
        )

    # Diagonal reference line (CKDI == BDRI)
    lim = max(merged["ckdi"].max(), merged["bdri"].max()) * 1.15
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5, label="CKDI = BDRI")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("CKDI")
    ax.set_ylabel("BDRI")
    ax.set_title("CKDI vs BDRI per Attack Class")
    ax.legend(fontsize=7)
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 4 – PCA 2-D projection
# ---------------------------------------------------------------------------

def plot_pca_projection(
    baseline: pd.DataFrame,
    attacks: dict[str, pd.DataFrame],
    feature_cols: list[str],
    out_path: str,
    max_samples: int = 500,
) -> None:
    """2-D PCA projection coloured by traffic class."""
    rng = np.random.default_rng(0)

    def _sample(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) > max_samples:
            idx = rng.choice(len(df), max_samples, replace=False)
            return df.iloc[idx]
        return df

    base_sample = _sample(baseline)
    parts = [base_sample]
    class_names = ["BENIGN"]
    for label, df_atk in attacks.items():
        parts.append(_sample(df_atk))
        class_names.append(label)

    combined = pd.concat(parts, ignore_index=True)
    X = combined[feature_cols].values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(7, 5))
    offset = 0
    for i, (name, df) in enumerate(zip(class_names, parts)):
        n = len(df)
        coords = X_pca[offset: offset + n]
        offset += n
        alpha = _BENIGN_ALPHA if name == "BENIGN" else _ATTACK_ALPHA
        marker = "o" if name == "BENIGN" else _ATTACK_MARKERS[(i - 1) % len(_ATTACK_MARKERS)]
        ax.scatter(
            coords[:, 0], coords[:, 1],
            s=15, alpha=alpha,
            color=_PALETTE[i % len(_PALETTE)],
            marker=marker,
            label=_wrap(name, 22),
        )

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} var.)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} var.)")
    ax.set_title("PCA Projection – Baseline vs Attack Classes")
    ax.legend(loc="best", markerscale=1.5, framealpha=0.7)
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 5 – Weight heatmap (SDI / ALR / RIF per class)
# ---------------------------------------------------------------------------

def plot_weight_heatmap(bdri_df: pd.DataFrame, out_path: str) -> None:
    """Heatmap of SLSQP-optimal weights (w_SDI, w_ALR, w_RIF) per attack class."""
    labels = bdri_df["attack_class"].tolist()
    weight_cols = ["w_sdi", "w_alr", "w_rif"]
    W = bdri_df[weight_cols].values  # shape (n_classes, 3)

    fig, ax = plt.subplots(figsize=(min(7, len(labels) * 1.1 + 2), 3.5))
    im = ax.imshow(W.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([_wrap(l) for l in labels], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(3))
    ax.set_yticklabels([r"$w_{\mathrm{SDI}}$", r"$w_{\mathrm{ALR}}$", r"$w_{\mathrm{RIF}}$"],
                       fontsize=9)
    ax.set_title("Optimal BDRI Weights per Attack Class (SLSQP)")

    # Annotate cells with values
    for j in range(len(labels)):
        for i in range(3):
            ax.text(j, i, f"{W[j, i]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if W[j, i] < 0.7 else "white")

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cb.set_label("Weight", fontsize=8)
    fig.tight_layout()
    _savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def save_all_figures(
    ckdi_df: pd.DataFrame,
    bdri_df: pd.DataFrame,
    baseline: pd.DataFrame,
    attacks: dict[str, pd.DataFrame],
    feature_cols: list[str],
    figures_dir: str = "figures",
) -> None:
    """Generate and save all five figures to *figures_dir*."""
    os.makedirs(figures_dir, exist_ok=True)
    plot_ckdi_bars(ckdi_df, os.path.join(figures_dir, "fig1_ckdi_bars.pdf"))
    plot_bdri_bars(bdri_df, os.path.join(figures_dir, "fig2_bdri_bars.pdf"))
    plot_ckdi_vs_bdri(ckdi_df, bdri_df, os.path.join(figures_dir, "fig3_ckdi_vs_bdri.pdf"))
    plot_pca_projection(
        baseline, attacks, feature_cols,
        os.path.join(figures_dir, "fig4_pca_projection.pdf"),
    )
    plot_weight_heatmap(bdri_df, os.path.join(figures_dir, "fig5_weight_heatmap.pdf"))


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------

def _wrap(text: str, max_len: int = 20) -> str:
    """Shorten long label strings for axis tick readability."""
    return text if len(text) <= max_len else text[:max_len - 1] + "…"
