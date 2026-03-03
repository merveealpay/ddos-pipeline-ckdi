#!/usr/bin/env python3
"""
run_pipeline.py – Single-command CKDI / BDRI pipeline for DDOS CICEV2023
=========================================================================

Usage
-----
    python run_pipeline.py                          # uses synthetic data
    python run_pipeline.py --data_dir Processed_Data
    python run_pipeline.py --data_dir /path/to/Processed_Data \\
                           --results_dir results_outputs \\
                           --figures_dir figures \\
                           --alpha 0.5

Outputs
-------
    results_outputs/ckdi_results.csv   – per-class CKDI with sub-components
    results_outputs/bdri_results.csv   – per-class BDRI with weights
    results_outputs/combined_summary.csv – CKDI + BDRI side by side
    figures/fig1_ckdi_bars.pdf
    figures/fig2_bdri_bars.pdf
    figures/fig3_ckdi_vs_bdri.pdf
    figures/fig4_pca_projection.pdf
    figures/fig5_weight_heatmap.pdf
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

# Ensure the project root is on sys.path when run as a script
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_dataset, split_by_label, feature_columns
from src.ckdi import compute_ckdi_detailed
from src.bdri import compute_bdri
from src.visualizer import save_all_figures

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CKDI / BDRI pipeline for DDOS CICEV2023.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        metavar="PATH",
        help="Path to the Processed_Data directory.  "
             "Omit to use the built-in synthetic dataset.",
    )
    parser.add_argument(
        "--results_dir",
        default="results_outputs",
        metavar="PATH",
        help="Directory for CSV output files.",
    )
    parser.add_argument(
        "--figures_dir",
        default="figures",
        metavar="PATH",
        help="Directory for PDF figure files.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="CKDI weight: alpha*Δ_stat + (1-alpha)*Δ_pca. Must be in [0,1].",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    # ── 1. Load data ──────────────────────────────────────────────────────
    logger.info("Step 1/4 – Loading dataset …")
    df = load_dataset(args.data_dir)
    baseline, attacks = split_by_label(df)
    feat_cols = feature_columns(df)

    logger.info(
        "Baseline samples: %d | Attack classes: %d | Features: %d",
        len(baseline), len(attacks), len(feat_cols),
    )

    # ── 2. CKDI ──────────────────────────────────────────────────────────
    logger.info("Step 2/4 – Computing CKDI (α=%.2f) …", args.alpha)
    ckdi_df = compute_ckdi_detailed(baseline, attacks, feat_cols, alpha=args.alpha)
    logger.info("CKDI results:\n%s", ckdi_df.to_string(index=False))

    # ── 3. BDRI ──────────────────────────────────────────────────────────
    logger.info("Step 3/4 – Computing BDRI (SLSQP weight optimisation) …")
    bdri_df = compute_bdri(baseline, attacks, feat_cols)
    logger.info("BDRI results:\n%s", bdri_df.to_string(index=False))

    # ── 4. Save outputs ───────────────────────────────────────────────────
    logger.info("Step 4/4 – Saving CSV outputs and figures …")

    os.makedirs(args.results_dir, exist_ok=True)

    ckdi_path = os.path.join(args.results_dir, "ckdi_results.csv")
    bdri_path = os.path.join(args.results_dir, "bdri_results.csv")
    summary_path = os.path.join(args.results_dir, "combined_summary.csv")

    ckdi_df.to_csv(ckdi_path, index=False)
    bdri_df.to_csv(bdri_path, index=False)

    # Combined summary: CKDI + BDRI merged on attack_class
    summary = ckdi_df[["attack_class", "delta_stat", "delta_pca", "ckdi"]].merge(
        bdri_df[["attack_class", "sdi", "alr", "rif",
                 "w_sdi", "w_alr", "w_rif", "weights_optimized", "bdri"]],
        on="attack_class",
    ).sort_values("ckdi", ascending=False).reset_index(drop=True)
    summary.to_csv(summary_path, index=False)

    logger.info("CSVs written to %r", args.results_dir)

    save_all_figures(
        ckdi_df=ckdi_df,
        bdri_df=bdri_df,
        baseline=baseline,
        attacks=attacks,
        feature_cols=feat_cols,
        figures_dir=args.figures_dir,
    )
    logger.info("Figures written to %r", args.figures_dir)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"  Results : {args.results_dir}/")
    print(f"  Figures : {args.figures_dir}/")
    print("=" * 60)
    print("\nCombined summary:")
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    run(args)
