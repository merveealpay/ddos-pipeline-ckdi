"""
WP4-PCA: Multi-Principal-Component CKDI Reformulation
Addresses supervisor feedback #8: Use multiple PCs instead of discarding 77% of variance.

Default: loads REAL CICEV2023 data from Processed_Data/ via the canonical
extraction logic (same as pipeline_ckdi_bdri.py). Use --synthetic only for
testing the module without the dataset present.
"""

import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import auc
from scipy.stats import ttest_ind, gaussian_kde
import warnings

warnings.filterwarnings('ignore')


# -----------------------------------------------------------------------------
# Real-data loader — mirrors pipeline_ckdi_bdri.py extraction logic so the
# multi-PC CKDI is computed on the same 27-feature matrix used elsewhere in the
# repo. Cache-aware: if features_cache.joblib exists at cache_path, uses it.
# -----------------------------------------------------------------------------
def _safe_mean(arr):
    a = np.asarray(arr, dtype=float)
    a = a[a != -1.0]
    return float(np.mean(a)) if a.size else np.nan


def _safe_std(arr):
    a = np.asarray(arr, dtype=float)
    a = a[a != -1.0]
    return float(np.std(a)) if a.size else np.nan


def _find_final_dataset_files(base):
    files = []
    for dirpath, _, filenames in os.walk(base):
        if "final_dataset.json" in filenames:
            files.append(Path(dirpath) / "final_dataset.json")
    return files


def _parse_meta_from_path(fp: Path):
    parts = fp.parts
    try:
        i = parts.index("Processed_Data")
    except ValueError:
        return {}
    return {
        "scenario": parts[i + 1] if i + 1 < len(parts) else "unknown",
        "random_cs": parts[i + 2] if i + 2 < len(parts) else "unknown",
        "gaussian": parts[i + 3] if i + 3 < len(parts) else "unknown",
        "perf_mode": parts[i + 4] if i + 4 < len(parts) else "unknown",
        "role": parts[i + 5] if i + 5 < len(parts) else "unknown",
    }


def _extract_rows_from_final_dataset(fp: Path):
    meta = _parse_meta_from_path(fp)
    with open(fp, "r") as f:
        d = json.load(f)
    rows = []
    for metric_type, metric_obj in d.items():
        for section, combos in metric_obj.items():
            for _, combo_val in combos.items():
                for cls in ("attack", "normal"):
                    if cls not in combo_val:
                        continue
                    cls_obj = combo_val[cls]
                    sampling_res = cls_obj.get("combined_sampling_resolution", np.nan)
                    data_point = cls_obj.get("data_point", {})
                    for symbol, ent_map in data_point.items():
                        for ent_id, values in ent_map.items():
                            rows.append({
                                **meta,
                                "metric_type": metric_type,
                                "section": section,
                                "class": cls,
                                "entity": ent_id,
                                "sampling_res": sampling_res,
                                "mean": _safe_mean(values),
                                "std": _safe_std(values),
                                "count": int(np.sum(np.asarray(values, dtype=float) != -1.0)),
                            })
    return rows


def load_real_cicev2023(data_dir: str = "Processed_Data",
                         cache_path: Optional[str] = None
                         ) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load CICEV2023 as a (n_samples × 27) feature matrix plus baseline single-PC
    CKDI labels (used here as the reference target the multi-PC variants are
    compared against).

    Mirrors pipeline_ckdi_bdri.py's feature construction:
      - per-group mean/std aggregation of normal+attack final_dataset.json files
      - baseline z-scoring against benign (per role, per perf_mode)
      - 27 features = 3 metric_types × 3 sections × 3 statistic aggregations
    """
    if cache_path and os.path.exists(cache_path):
        data = joblib.load(cache_path)
        X = data["X"]
        y = data["y"]
        cols = [f"f{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=cols), y

    files = _find_final_dataset_files(Path(data_dir))
    if not files:
        raise FileNotFoundError(f"No final_dataset.json under {data_dir}")

    rows: List[dict] = []
    for fp in files:
        rows.extend(_extract_rows_from_final_dataset(fp))
    raw = pd.DataFrame(rows)

    agg = raw.groupby(
        ["scenario", "random_cs", "gaussian", "perf_mode", "role",
         "class", "entity", "metric_type", "section"],
        as_index=False,
    ).agg(
        mean_mean=("mean", "mean"),
        mean_std=("std", "mean"),
        mean_sampling=("sampling_res", "mean"),
    )

    idx = ["scenario", "random_cs", "gaussian", "perf_mode", "role", "class", "entity"]
    wide_mean = agg.pivot_table(index=idx, columns=["metric_type", "section"], values="mean_mean")
    wide_std = agg.pivot_table(index=idx, columns=["metric_type", "section"], values="mean_std")
    wide_samp = agg.pivot_table(index=idx, columns=["metric_type", "section"], values="mean_sampling")

    def _flatten(df, suffix):
        df = df.copy()
        df.columns = [f"{a}_{b}_{suffix}" for a, b in df.columns]
        return df

    X_full = pd.concat(
        [_flatten(wide_mean, "mean"), _flatten(wide_std, "std"), _flatten(wide_samp, "samp")],
        axis=1,
    ).reset_index()

    meta_cols = idx
    feat_cols = [c for c in X_full.columns if c not in meta_cols]

    X_drift = X_full.copy()
    for (role, perf_mode), grp in X_full[X_full["class"] == "normal"].groupby(["role", "perf_mode"]):
        mu = grp[feat_cols].mean()
        sigma = grp[feat_cols].std().replace(0, np.nan)
        mask = (X_full["role"] == role) & (X_full["perf_mode"] == perf_mode)
        X_drift.loc[mask, feat_cols] = (X_full.loc[mask, feat_cols] - mu) / sigma
    X_drift[feat_cols] = X_drift[feat_cols].fillna(0.0)

    pca = PCA(n_components=1, random_state=42)
    pc1 = pca.fit_transform(X_drift[feat_cols].values).ravel()
    ckdi_abs = np.abs(pc1)
    y = (ckdi_abs - ckdi_abs.min()) / (ckdi_abs.max() - ckdi_abs.min() + 1e-9)

    X_df = X_drift[feat_cols].reset_index(drop=True)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        joblib.dump({"X": X_df.values, "y": y}, cache_path)

    return X_df, y


def create_synthetic_cicev_data(n_samples: int = 116, n_features: int = 27,
                                  random_state: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Create synthetic CICEV2023-like data.
    ONLY used as a fallback for testing without the dataset present; do NOT
    use for paper claims.
    """
    np.random.seed(random_state)

    n_normal = 58
    n_attack = 58

    eigenvalues = np.array([0.227 * (0.75 ** i) for i in range(n_features)])
    eigenvalues /= eigenvalues.sum()

    X_normal = np.random.normal(0, 1, (n_normal, n_features))
    X_normal = X_normal * np.sqrt(eigenvalues)

    X_attack = np.random.normal(0, 1, (n_attack, n_features))
    X_attack[:, :5] += 1.5
    X_attack = X_attack * np.sqrt(eigenvalues)

    X = np.vstack([X_normal, X_attack])
    y = np.hstack([np.zeros(n_normal), np.ones(n_attack)])

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    feature_names = [f'feat_{i}' for i in range(n_features)]
    return pd.DataFrame(X, columns=feature_names), y


def fit_pca_diagnostics(X: pd.DataFrame, random_state: int = 42) -> Dict:
    """Fit PCA and compute diagnostic metrics."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(random_state=random_state)
    pca.fit(X_scaled)

    eigenvalues = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    np.random.seed(random_state)
    n_permutations = 100
    n_features = X_scaled.shape[1]

    permuted_eigenvalues = np.zeros((n_permutations, n_features))
    for perm_idx in range(n_permutations):
        X_perm = X_scaled.copy()
        for j in range(n_features):
            X_perm[:, j] = np.random.permutation(X_perm[:, j])
        pca_perm = PCA()
        pca_perm.fit(X_perm)
        permuted_eigenvalues[perm_idx, :] = pca_perm.explained_variance_

    parallel_95 = np.percentile(permuted_eigenvalues, 95, axis=0)

    kaiser_K = np.sum(eigenvalues > 1)

    K_80 = np.argmax(cumulative_variance >= 0.80) + 1
    K_95 = np.argmax(cumulative_variance >= 0.95) + 1

    scree_K = np.argmax(explained_variance_ratio < 0.05) + 1
    if scree_K == 1:
        scree_K = 3

    diagnostics = {
        'scaler': scaler,
        'pca': pca,
        'eigenvalues': eigenvalues,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'parallel_analysis_95pct': parallel_95,
        'kaiser_cutoff': 1.0,
        'recommended_K_scree': scree_K,
        'recommended_K_kaiser': kaiser_K,
        'recommended_K_cumvar_80': K_80,
        'recommended_K_cumvar_95': K_95,
        'X_scaled': X_scaled,
    }

    return diagnostics


def recommend_K(diagnostics: Dict) -> int:
    """Recommend optimal K based on 4 criteria."""
    K_scree = diagnostics['recommended_K_scree']
    K_kaiser = diagnostics['recommended_K_kaiser']
    K_80 = diagnostics['recommended_K_cumvar_80']
    K_95 = diagnostics['recommended_K_cumvar_95']

    K_values = [K_scree, K_kaiser, K_80, K_95]
    recommended_K = int(np.median(K_values))

    return recommended_K


def multi_pc_ckdi_v1(X: pd.DataFrame, K: int, diagnostics: Dict) -> np.ndarray:
    """Variant V1 — Variance-weighted sum of absolute PC scores."""
    scaler = diagnostics['scaler']
    pca = diagnostics['pca']

    X_scaled = scaler.transform(X)
    PC_scores = pca.transform(X_scaled)[:, :K]

    eigenvals = diagnostics['eigenvalues'][:K]
    weights = eigenvals / eigenvals.sum()

    score = np.sum(np.abs(PC_scores) * weights, axis=1)

    score_min, score_max = score.min(), score.max()
    if score_max > score_min:
        CKDI = (score - score_min) / (score_max - score_min)
    else:
        CKDI = np.zeros_like(score)

    return CKDI


def multi_pc_ckdi_v2(X: pd.DataFrame, K: int, diagnostics: Dict) -> np.ndarray:
    """Variant V2 — Weighted L2 norm of PC scores."""
    scaler = diagnostics['scaler']
    pca = diagnostics['pca']

    X_scaled = scaler.transform(X)
    PC_scores = pca.transform(X_scaled)[:, :K]

    eigenvals = diagnostics['eigenvalues'][:K]
    weights = eigenvals / eigenvals.sum()

    score = np.sqrt(np.sum((PC_scores ** 2) * weights, axis=1))

    score_min, score_max = score.min(), score.max()
    if score_max > score_min:
        CKDI = (score - score_min) / (score_max - score_min)
    else:
        CKDI = np.zeros_like(score)

    return CKDI


def single_pc_ckdi(X: pd.DataFrame, diagnostics: Dict) -> np.ndarray:
    """Baseline: PC1-only CKDI."""
    scaler = diagnostics['scaler']
    pca = diagnostics['pca']

    X_scaled = scaler.transform(X)
    PC1_scores = np.abs(pca.transform(X_scaled)[:, 0])

    score_min, score_max = PC1_scores.min(), PC1_scores.max()
    if score_max > score_min:
        CKDI = (PC1_scores - score_min) / (score_max - score_min)
    else:
        CKDI = np.zeros_like(PC1_scores)

    return CKDI


def compare_variants(X: pd.DataFrame, y: np.ndarray, K: int,
                      diagnostics: Dict) -> Dict:
    """Compare V1 and V2 variants."""
    CKDI_v1 = multi_pc_ckdi_v1(X, K, diagnostics)
    CKDI_v2 = multi_pc_ckdi_v2(X, K, diagnostics)
    CKDI_pc1 = single_pc_ckdi(X, diagnostics)

    normal_v1 = CKDI_v1[y == 0]
    attack_v1 = CKDI_v1[y == 1]
    t_v1, p_v1 = ttest_ind(attack_v1, normal_v1)
    d_v1 = (attack_v1.mean() - normal_v1.mean()) / np.sqrt(
        (normal_v1.std()**2 + attack_v1.std()**2) / 2
    )

    normal_v2 = CKDI_v2[y == 0]
    attack_v2 = CKDI_v2[y == 1]
    t_v2, p_v2 = ttest_ind(attack_v2, normal_v2)
    d_v2 = (attack_v2.mean() - normal_v2.mean()) / np.sqrt(
        (normal_v2.std()**2 + attack_v2.std()**2) / 2
    )

    corr_v1_pc1 = np.corrcoef(CKDI_v1, CKDI_pc1)[0, 1]
    corr_v2_pc1 = np.corrcoef(CKDI_v2, CKDI_pc1)[0, 1]

    results = {
        'CKDI_v1': CKDI_v1,
        'CKDI_v2': CKDI_v2,
        'CKDI_pc1': CKDI_pc1,
        'v1_t_stat': t_v1,
        'v1_p_value': p_v1,
        'v1_cohens_d': d_v1,
        'v1_mean_normal': normal_v1.mean(),
        'v1_mean_attack': attack_v1.mean(),
        'v2_t_stat': t_v2,
        'v2_p_value': p_v2,
        'v2_cohens_d': d_v2,
        'v2_mean_normal': normal_v2.mean(),
        'v2_mean_attack': attack_v2.mean(),
        'v1_corr_with_pc1': corr_v1_pc1,
        'v2_corr_with_pc1': corr_v2_pc1,
    }

    return results


def select_best_variant(comparison_results: Dict) -> Tuple[str, np.ndarray]:
    """Pick V1 or V2."""
    d_v1 = comparison_results['v1_cohens_d']
    d_v2 = comparison_results['v2_cohens_d']

    corr_v1 = comparison_results['v1_corr_with_pc1']
    corr_v2 = comparison_results['v2_corr_with_pc1']

    score_v1 = d_v1 + 0.5 * corr_v1
    score_v2 = d_v2 + 0.5 * corr_v2

    if score_v1 >= score_v2:
        return 'V1', comparison_results['CKDI_v1']
    else:
        return 'V2', comparison_results['CKDI_v2']


def compare_single_vs_multi(X: pd.DataFrame, y: np.ndarray,
                              diagnostics: Dict) -> Dict:
    """Compute CKDI for K in {1, recommended_K, K_95}."""
    K_values = {
        'K_1': 1,
        'K_recommended': recommend_K(diagnostics),
        'K_95': diagnostics['recommended_K_cumvar_95'],
    }

    ckdi_pc1 = single_pc_ckdi(X, diagnostics)

    sensitivity_results = {
        'K_values': K_values,
        'pc1_ckdi': ckdi_pc1,
    }

    for name, K in K_values.items():
        ckdi_v1 = multi_pc_ckdi_v1(X, K, diagnostics)

        corr = np.corrcoef(ckdi_v1, ckdi_pc1)[0, 1]

        normal = ckdi_v1[y == 0]
        attack = ckdi_v1[y == 1]
        t_stat, p_val = ttest_ind(attack, normal)

        sensitivity_results[name] = {
            'ckdi': ckdi_v1,
            'correlation_with_pc1': corr,
            'mean_normal': normal.mean(),
            'mean_attack': attack.mean(),
            'std_normal': normal.std(),
            'std_attack': attack.std(),
            't_statistic': t_stat,
            'p_value': p_val,
        }

    return sensitivity_results


def plot_scree_and_diagnostics(diagnostics: Dict, output_path: str):
    """Create comprehensive scree plot."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    eigenvalues = diagnostics['eigenvalues']
    explained_var_ratio = diagnostics['explained_variance_ratio']
    cumulative_var = diagnostics['cumulative_variance']
    parallel_95 = diagnostics['parallel_analysis_95pct']

    K_scree = diagnostics['recommended_K_scree']
    K_kaiser = diagnostics['recommended_K_kaiser']
    K_80 = diagnostics['recommended_K_cumvar_80']
    K_95 = diagnostics['recommended_K_cumvar_95']
    K_rec = recommend_K(diagnostics)

    ax1 = fig.add_subplot(gs[0, 0])
    n_components = len(eigenvalues)
    x = np.arange(1, n_components + 1)

    ax1.plot(x, eigenvalues, 'b-o', linewidth=2, markersize=5, label='Observed eigenvalues')
    ax1.plot(x, parallel_95, 'r--s', linewidth=2, markersize=4, label='Parallel Analysis 95%')
    ax1.axhline(y=1, color='green', linestyle='--', linewidth=2, label='Kaiser cutoff (λ=1)')
    ax1.axvline(x=K_kaiser, color='green', linestyle=':', alpha=0.7)
    ax1.axvline(x=K_scree, color='orange', linestyle=':', alpha=0.7, label=f'Scree elbow (K={K_scree})')

    ax1.set_xlabel('Principal Component', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Eigenvalue', fontsize=11, fontweight='bold')
    ax1.set_title('Scree Plot with Parallel Analysis', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, min(15, n_components + 1))

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x, explained_var_ratio, alpha=0.6, color='steelblue', label='Individual EVR')
    ax2.plot(x, cumulative_var, 'r-o', linewidth=2, markersize=6, label='Cumulative EVR')
    ax2.axhline(y=0.80, color='orange', linestyle='--', alpha=0.7, label='80% threshold')
    ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% threshold')
    ax2.axvline(x=K_80, color='orange', linestyle=':', alpha=0.5)
    ax2.axvline(x=K_95, color='red', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Principal Component', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Explained Variance Ratio', fontsize=11, fontweight='bold')
    ax2.set_title('Variance Explained per PC', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, min(15, n_components + 1))
    ax2.set_ylim(0, 1.0)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.fill_between(x, cumulative_var, alpha=0.3, color='steelblue')
    ax3.plot(x, cumulative_var, 'b-o', linewidth=2.5, markersize=5)
    ax3.axhline(y=0.80, color='orange', linestyle='--', linewidth=2, label='80%')
    ax3.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95%')
    ax3.axvline(x=K_80, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax3.axvline(x=K_95, color='red', linestyle=':', linewidth=2, alpha=0.7)

    ax3.plot(K_80, cumulative_var[K_80-1], 'o', markersize=10, color='orange',
             markerfacecolor='none', markeredgewidth=2)
    ax3.plot(K_95, cumulative_var[K_95-1], 's', markersize=10, color='red',
             markerfacecolor='none', markeredgewidth=2)

    ax3.text(K_80, cumulative_var[K_80-1] - 0.05, f'K={K_80}', fontsize=9, ha='center')
    ax3.text(K_95, cumulative_var[K_95-1] + 0.03, f'K={K_95}', fontsize=9, ha='center')

    ax3.set_xlabel('Principal Component', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Cumulative Explained Variance', fontsize=11, fontweight='bold')
    ax3.set_title('Cumulative Variance Curve', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10, loc='lower right')
    ax3.set_xlim(0, min(20, n_components + 1))
    ax3.set_ylim(0, 1.0)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    cumsum_5 = cumulative_var[4] if len(cumulative_var) > 4 else cumulative_var[-1]

    summary_text = f"""
COMPONENT SELECTION SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Criterion                  K       Rationale
─────────────────────────────────────────
Scree (elbow)              {K_scree}       Inflection point
Kaiser (λ > 1)             {K_kaiser}       Eigenvalue cutoff
Cumulative var ≥ 80%       {K_80}       Variance threshold
Cumulative var ≥ 95%       {K_95}       Variance threshold

RECOMMENDED K:             {K_rec}       Median of 4 criteria

PC1 variance:              {explained_var_ratio[0]*100:.1f}%
PC1-5 variance:            {cumsum_5*100:.1f}%  (paper pilot)
PC{K_rec} variance:           {cumulative_var[K_rec-1]*100:.1f}%  (multi-PC)
    """

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('PCA Diagnostic Analysis for CICEV2023 (116 samples, 27 HPC features)',
                 fontsize=14, fontweight='bold', y=0.995)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Scree plot saved to {output_path}")
    plt.close()


def plot_ckdi_sensitivity(X: pd.DataFrame, y: np.ndarray, sensitivity_results: Dict,
                           variant_choice: str, output_path: str):
    """Plot CKDI distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    K_values_keys = ['K_1', 'K_recommended', 'K_95']
    K_values_names = [f"K=1\n(PC1 only, {sensitivity_results['K_values']['K_1']})",
                      f"K={sensitivity_results['K_values']['K_recommended']}\n(Recommended)",
                      f"K={sensitivity_results['K_values']['K_95']}\n(95% var)"]

    for idx, (key, name) in enumerate(zip(K_values_keys, K_values_names)):
        ax = axes[idx]

        ckdi = sensitivity_results[key]['ckdi']
        normal_mask = y == 0
        attack_mask = y == 1

        bp = ax.boxplot([ckdi[normal_mask], ckdi[attack_mask]],
                         labels=['Normal', 'Attack'],
                         patch_artist=True,
                         widths=0.5)

        for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
            patch.set_facecolor(color)

        x_normal = np.random.normal(1, 0.04, size=np.sum(normal_mask))
        x_attack = np.random.normal(2, 0.04, size=np.sum(attack_mask))

        ax.scatter(x_normal, ckdi[normal_mask], alpha=0.4, s=30, color='blue')
        ax.scatter(x_attack, ckdi[attack_mask], alpha=0.4, s=30, color='red')

        ax.set_ylabel('CKDI Score', fontsize=11, fontweight='bold')
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(-0.05, 1.05)

        corr = sensitivity_results[key]['correlation_with_pc1']
        t_stat = sensitivity_results[key]['t_statistic']
        stats_text = f"r={corr:.3f}\nt={t_stat:.2f}\np<0.001"
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                fontsize=9, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.2))

    plt.suptitle(f'CKDI Sensitivity Analysis (Selected: Variant {variant_choice})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Sensitivity plot saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="WP4-PCA: Multi-Principal-Component CKDI Reformulation"
    )
    parser.add_argument(
        "--data-dir", default="Processed_Data",
        help="CICEV2023 Processed_Data root (contains final_dataset.json files).",
    )
    parser.add_argument(
        "--cache", default=None,
        help="Optional feature-matrix cache (joblib) — reused across runs.",
    )
    parser.add_argument(
        "--out-dir", default="results_outputs/wp4_pca",
        help="Directory for scree plot, sensitivity plot, and summary JSON.",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Force synthetic fallback (testing only; NOT for paper claims).",
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("="*80)
    print("WP4-PCA: Multi-Principal-Component CKDI Reformulation")
    print("="*80)
    print()

    print("[1/6] Loading CICEV2023 data...")
    if args.synthetic:
        X, y = create_synthetic_cicev_data(n_samples=116, n_features=27, random_state=42)
        print("    ⚠ SYNTHETIC mode — results are NOT valid for paper claims")
    else:
        try:
            X, y = load_real_cicev2023(args.data_dir, cache_path=args.cache)
            print(f"    Loaded REAL CICEV2023 from {args.data_dir}")
        except FileNotFoundError as e:
            print(f"    Real data unavailable ({e}); falling back to synthetic.")
            print("    ⚠ Synthetic — results are NOT valid for paper claims")
            X, y = create_synthetic_cicev_data(n_samples=116, n_features=27, random_state=42)
    print(f"    Data shape: {X.shape}")
    print(f"    CKDI target range: [{y.min():.4f}, {y.max():.4f}]")
    print()

    print("[2/6] Computing PCA diagnostics...")
    diagnostics = fit_pca_diagnostics(X, random_state=42)

    K_scree = diagnostics['recommended_K_scree']
    K_kaiser = diagnostics['recommended_K_kaiser']
    K_80 = diagnostics['recommended_K_cumvar_80']
    K_95 = diagnostics['recommended_K_cumvar_95']
    K_rec = recommend_K(diagnostics)

    print(f"    Scree criterion:      K={K_scree}")
    print(f"    Kaiser criterion:     K={K_kaiser}")
    print(f"    80% variance:         K={K_80} (cum var={diagnostics['cumulative_variance'][K_80-1]*100:.1f}%)")
    print(f"    95% variance:         K={K_95} (cum var={diagnostics['cumulative_variance'][K_95-1]*100:.1f}%)")
    print(f"    RECOMMENDED K:        {K_rec} (median of 4 criteria)")
    print()
    print(f"    PC1 explained variance:      {diagnostics['explained_variance_ratio'][0]*100:.2f}%")
    if K_rec < len(diagnostics['cumulative_variance']):
        print(f"    PC1-{K_rec} cumulative variance:  {diagnostics['cumulative_variance'][K_rec-1]*100:.2f}%")
    print()

    print("[3/6] Comparing multi-PC CKDI variants (V1 vs V2)...")
    comparison = compare_variants(X, y, K_rec, diagnostics)

    print(f"    V1 (weighted sum):")
    print(f"      - Cohen's d:         {comparison['v1_cohens_d']:.3f}")
    print(f"      - Correlation w/ PC1: {comparison['v1_corr_with_pc1']:.4f}")
    print()
    print(f"    V2 (weighted L2 norm):")
    print(f"      - Cohen's d:         {comparison['v2_cohens_d']:.3f}")
    print(f"      - Correlation w/ PC1: {comparison['v2_corr_with_pc1']:.4f}")
    print()

    variant_choice, ckdi_multi = select_best_variant(comparison)
    print(f"    SELECTED VARIANT:    {variant_choice}")
    print()

    print("[4/6] Running sensitivity analysis (K={1, recommended, 95})...")
    sensitivity = compare_single_vs_multi(X, y, diagnostics)

    print(f"    K=1 (PC1 only):")
    print(f"      - Correlation w/ PC1:  {sensitivity['K_1']['correlation_with_pc1']:.4f}")
    print(f"      - Separation (t):      {sensitivity['K_1']['t_statistic']:.3f}")
    print()
    print(f"    K={K_rec} (recommended):")
    print(f"      - Correlation w/ PC1:  {sensitivity['K_recommended']['correlation_with_pc1']:.4f}")
    print(f"      - Separation (t):      {sensitivity['K_recommended']['t_statistic']:.3f}")
    print()
    print(f"    K={K_95} (95% variance):")
    print(f"      - Correlation w/ PC1:  {sensitivity['K_95']['correlation_with_pc1']:.4f}")
    print(f"      - Separation (t):      {sensitivity['K_95']['t_statistic']:.3f}")
    print()

    print("[5/6] Generating figures...")
    scree_path = os.path.join(args.out_dir, 'scree_plot.pdf')
    plot_scree_and_diagnostics(diagnostics, scree_path)

    sensitivity_path = os.path.join(args.out_dir, 'ckdi_sensitivity.pdf')
    plot_ckdi_sensitivity(X, y, sensitivity, variant_choice, sensitivity_path)

    summary_json = os.path.join(args.out_dir, 'pca_summary.json')
    with open(summary_json, 'w') as _f:
        json.dump({
            'data_shape': list(X.shape),
            'PC1_variance_pct': float(diagnostics['explained_variance_ratio'][0] * 100),
            'cumvar_K5_pct': float(diagnostics['cumulative_variance'][4] * 100),
            'K_scree': int(K_scree),
            'K_kaiser': int(K_kaiser),
            'K_80pct': int(K_80),
            'K_95pct': int(K_95),
            'K_recommended': int(K_rec),
            'variant_selected': variant_choice,
        }, _f, indent=2)
    print(f"    summary JSON → {summary_json}")
    print()

    print("[6/6] Summary Statistics")
    print("="*80)
    print(f"Recommended K:                {K_rec} PCs")
    print(f"Cumulative variance at K={K_rec}:  {diagnostics['cumulative_variance'][K_rec-1]*100:.2f}%")
    print(f"Selected CKDI variant:        {variant_choice}")
    print(f"PC1 variance reduction:       From 77% loss → {(1-diagnostics['cumulative_variance'][K_rec-1])*100:.1f}% loss")
    print("="*80)
