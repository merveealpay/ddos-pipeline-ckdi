#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import joblib

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def safe_mean(arr):
    a = np.asarray(arr, dtype=float)
    mask = (a != -1.0)
    a = a[mask]
    return float(np.mean(a)) if a.size else np.nan

def safe_std(arr):
    a = np.asarray(arr, dtype=float)
    mask = (a != -1.0)
    a = a[mask]
    return float(np.std(a)) if a.size else np.nan

def find_final_dataset_files(base):
    files = []
    for dirpath, _, filenames in os.walk(base):
        if "final_dataset.json" in filenames:
            files.append(Path(dirpath) / "final_dataset.json")
    return files

def parse_meta_from_path(fp):
    parts = fp.parts
    try:
        i = parts.index("Processed_Data")
    except ValueError:
        return {}
    return {
        "scenario": parts[i+1] if i+1 < len(parts) else "unknown",
        "random_cs": parts[i+2] if i+2 < len(parts) else "unknown",
        "gaussian": parts[i+3] if i+3 < len(parts) else "unknown",
        "perf_mode": parts[i+4] if i+4 < len(parts) else "unknown",
        "role": parts[i+5] if i+5 < len(parts) else "unknown",
    }

def extract_rows_from_final_dataset(fp):
    meta = parse_meta_from_path(fp)
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
                                "mean": safe_mean(values),
                                "std": safe_std(values),
                                "count": int(np.sum(np.asarray(values, dtype=float) != -1.0)),
                            })
    return rows

def extract_features_and_ckdi(data_dir, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        data = joblib.load(cache_path)
        return data['X'], data['y']

    print(f"Extracting features from {data_dir}...")
    final_files = find_final_dataset_files(Path(data_dir))
    print(f"Found {len(final_files)} final_dataset.json files")

    all_rows = []
    for fp in final_files:
        all_rows.extend(extract_rows_from_final_dataset(fp))

    raw_df = pd.DataFrame(all_rows)
    print(f"Extracted {len(raw_df)} raw rows")

    agg = raw_df.groupby(
        ["scenario", "random_cs", "gaussian", "perf_mode", "role", "class", "entity", "metric_type", "section"],
        as_index=False
    ).agg(
        mean_mean=("mean", "mean"),
        mean_std=("std", "mean"),
        mean_sampling=("sampling_res", "mean"),
    )

    wide_mean = agg.pivot_table(
        index=["scenario", "random_cs", "gaussian", "perf_mode", "role", "class", "entity"],
        columns=["metric_type", "section"],
        values="mean_mean",
    )
    wide_std = agg.pivot_table(
        index=["scenario", "random_cs", "gaussian", "perf_mode", "role", "class", "entity"],
        columns=["metric_type", "section"],
        values="mean_std",
    )
    wide_samp = agg.pivot_table(
        index=["scenario", "random_cs", "gaussian", "perf_mode", "role", "class", "entity"],
        columns=["metric_type", "section"],
        values="mean_sampling",
    )

    def flatten(df, suffix):
        df = df.copy()
        df.columns = [f"{a}_{b}_{suffix}" for a, b in df.columns]
        return df

    X = pd.concat([flatten(wide_mean, "mean"), flatten(wide_std, "std"), flatten(wide_samp, "samp")], axis=1).reset_index()

    meta_cols = ["scenario", "random_cs", "gaussian", "perf_mode", "role", "class", "entity"]
    feat_cols = [c for c in X.columns if c not in meta_cols]

    print(f"Feature set shape: {len(X)} samples x {len(feat_cols)} features")

    X_drift = X.copy()
    for (role, perf_mode), grp in X[X["class"] == "normal"].groupby(["role", "perf_mode"]):
        mu = grp[feat_cols].mean()
        sigma = grp[feat_cols].std().replace(0, np.nan)
        mask = (X["role"] == role) & (X["perf_mode"] == perf_mode)
        X_drift.loc[mask, feat_cols] = (X.loc[mask, feat_cols] - mu) / sigma

    X_drift[feat_cols] = X_drift[feat_cols].fillna(0.0)

    pca = PCA(n_components=1, random_state=42)
    pc1 = pca.fit_transform(X_drift[feat_cols].values).ravel()
    ckdi_abs = np.abs(pc1)
    y = (ckdi_abs - ckdi_abs.min()) / (ckdi_abs.max() - ckdi_abs.min() + 1e-9)

    X_features = X_drift[feat_cols].values

    print(f"CKDI range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"Features shape: {X_features.shape}")

    if cache_path:
        cache_dir = os.path.dirname(cache_path)
        os.makedirs(cache_dir, exist_ok=True)
        joblib.dump({'X': X_features, 'y': y}, cache_path)
        print(f"Cached to {cache_path}")

    return X_features, y

def train_ensemble(X, y, seed=42, output_dir=None):
    np.random.seed(seed)

    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost required")

    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=seed)
    n_folds = 15

    predictions = {
        'xgboost': np.zeros_like(y),
        'random_forest': np.zeros_like(y),
        'mlp': np.zeros_like(y),
        'ensemble': np.zeros_like(y),
    }

    fold_idx = 0

    print("Starting 5-fold x 3-repeat CV...")
    for train_idx, val_idx in cv.split(X):
        fold_idx += 1
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        xgb = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=1.0,
            random_state=seed,
            verbosity=0,
        )
        xgb.fit(X_train, y_train)
        y_xgb = xgb.predict(X_val)

        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=seed,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        y_rf = rf.predict(X_val)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=False,
            random_state=seed,
            verbose=0,
        )
        mlp.fit(X_train_scaled, y_train)
        y_mlp = mlp.predict(X_val_scaled)

        y_ensemble = (y_xgb + y_rf + y_mlp) / 3.0

        predictions['xgboost'][val_idx] = y_xgb
        predictions['random_forest'][val_idx] = y_rf
        predictions['mlp'][val_idx] = y_mlp
        predictions['ensemble'][val_idx] = y_ensemble

        if fold_idx % 5 == 0:
            print(f"  Fold {fold_idx}/{n_folds} complete")

    print(f"CV completed. {n_folds} folds processed.")

    results = {}
    for model_name in ['xgboost', 'random_forest', 'mlp', 'ensemble']:
        y_pred = predictions[model_name]
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        results[model_name] = {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
        }
        print(f"{model_name:15} | R2 = {r2:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}")

    print("Training final models on full dataset...")

    xgb_final = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=1.0,
        random_state=seed,
        verbosity=0,
    )
    xgb_final.fit(X, y)

    rf_final = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=seed,
        n_jobs=-1,
    )
    rf_final.fit(X, y)

    scaler_final = StandardScaler()
    X_scaled_final = scaler_final.fit_transform(X)

    mlp_final = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        early_stopping=False,
        random_state=seed,
        verbose=0,
    )
    mlp_final.fit(X_scaled_final, y)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(xgb_final, output_dir / 'xgb.joblib')
        joblib.dump(rf_final, output_dir / 'rf.joblib')
        joblib.dump(mlp_final, output_dir / 'mlp.joblib')
        joblib.dump(scaler_final, output_dir / 'scaler.joblib')
        print(f"Models saved to {output_dir}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Ensemble Severity Regression")
    parser.add_argument('--data-dir', default='/tmp/ddos-repo/Processed_Data')
    parser.add_argument('--output-dir', default='/sessions/youthful-cool-carson/mnt/outputs/stream_c/models')
    parser.add_argument('--results-dir', default='/sessions/youthful-cool-carson/mnt/outputs/stream_c/results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cache', default='/sessions/youthful-cool-carson/mnt/outputs/stream_c/features_cache.joblib')

    args = parser.parse_args()

    print("=" * 80)
    print("ENSEMBLE SEVERITY REGRESSION (III-D)")
    print("=" * 80)
    print(f"Data dir: {args.data_dir}")
    print(f"Seed: {args.seed}")
    print()

    X, y = extract_features_and_ckdi(args.data_dir, cache_path=args.cache)
    results = train_ensemble(X, y, seed=args.seed, output_dir=args.output_dir)

    output_path = Path(args.results_dir) / 'ensemble_metrics.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_json = {
        'xgboost': {
            'r2': float(results['xgboost']['r2']),
            'mae': float(results['xgboost']['mae']),
            'rmse': float(results['xgboost']['rmse']),
        },
        'random_forest': {
            'r2': float(results['random_forest']['r2']),
            'mae': float(results['random_forest']['mae']),
            'rmse': float(results['random_forest']['rmse']),
        },
        'mlp': {
            'r2': float(results['mlp']['r2']),
            'mae': float(results['mlp']['mae']),
            'rmse': float(results['mlp']['rmse']),
        },
        'ensemble': {
            'r2': float(results['ensemble']['r2']),
            'mae': float(results['ensemble']['mae']),
            'rmse': float(results['ensemble']['rmse']),
        },
        'paper_reference': {
            'xgboost_r2': 0.614,
            'random_forest_r2': 0.537,
            'mlp_r2': 0.663,
            'ensemble_r2': 0.765,
        }
    }

    with open(output_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)

    print("=" * 80)
    print(f"Results saved to: {output_path}")
    print("=" * 80)

if __name__ == '__main__':
    main()
