#\!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
import joblib

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def extract_features_and_ckdi(data_dir, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        data = joblib.load(cache_path)
        return data['X'], data['y']
    return None, None

def train_ensemble_tuned(X, y, seed=42, output_dir=None):
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost required")
    np.random.seed(seed)
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=seed)
    n_folds = 15
    predictions = {'xgboost': np.zeros_like(y), 'random_forest': np.zeros_like(y), 'mlp': np.zeros_like(y), 'ensemble': np.zeros_like(y)}
    fold_idx = 0
    print("Starting 5-fold x 3-repeat CV with tuned MLP...")
    for train_idx, val_idx in cv.split(X):
        fold_idx += 1
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        xgb = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=1.0, random_state=seed, verbosity=0)
        xgb.fit(X_train, y_train)
        y_xgb = xgb.predict(X_val)
        rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=seed, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_rf = rf.predict(X_val)
        mlp_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='lbfgs',
                alpha=0.006796578090758156, learning_rate='constant', learning_rate_init=0.00011527987128232407,
                max_iter=1031, early_stopping=False, batch_size=64, random_state=seed, verbose=0))
        ])
        mlp_pipeline.fit(X_train, y_train)
        y_mlp = mlp_pipeline.predict(X_val)
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
        results[model_name] = {'r2': r2, 'mae': mae, 'rmse': rmse}
        print(f"{model_name:15} | R2 = {r2:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Ensemble Regression with Tuned MLP")
    parser.add_argument('--data-dir', default='/tmp/ddos-repo/Processed_Data')
    parser.add_argument('--output-dir', default='/sessions/youthful-cool-carson/mnt/outputs/stream_f/models')
    parser.add_argument('--results-dir', default='/sessions/youthful-cool-carson/mnt/outputs/stream_f/results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cache', default='/sessions/youthful-cool-carson/mnt/outputs/stream_b_real/features_cache.joblib')
    args = parser.parse_args()
    print("=" * 80)
    print("ENSEMBLE REGRESSION WITH TUNED MLP (STREAM F)")
    print("=" * 80)
    X, y = extract_features_and_ckdi(args.data_dir, cache_path=args.cache)
    results = train_ensemble_tuned(X, y, seed=args.seed, output_dir=args.output_dir)
    output_path = Path(args.results_dir) / 'ensemble_metrics_tuned.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_json = {
        'xgboost': {'r2': float(results['xgboost']['r2']), 'mae': float(results['xgboost']['mae']), 'rmse': float(results['xgboost']['rmse'])},
        'random_forest': {'r2': float(results['random_forest']['r2']), 'mae': float(results['random_forest']['mae']), 'rmse': float(results['random_forest']['rmse'])},
        'mlp': {'r2': float(results['mlp']['r2']), 'mae': float(results['mlp']['mae']), 'rmse': float(results['mlp']['rmse'])},
        'ensemble': {'r2': float(results['ensemble']['r2']), 'mae': float(results['ensemble']['mae']), 'rmse': float(results['ensemble']['rmse'])},
    }
    with open(output_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Results saved to: {output_path}")

if __name__ == '__main__':
    main()
