#\!/usr/bin/env python3
"""
Stream F: Optuna Bayesian Hyperparameter Optimization for MLP
Objective: Close gap between Stream C's R²=0.128 and paper's claimed R²=0.6631

Strategy:
- Load features cache (116 samples x 27 features)
- RepeatedKFold (5 splits x 3 repeats = 15 folds, seed=42)
- Optuna Bayesian search over 50 trials
- Keep architecture fixed (128-64-32), tune hyperparams
- Target: maximize mean R² across 15 CV folds
"""

import argparse
import json
import os
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

SEED = 42
CACHE_PATH = '/sessions/youthful-cool-carson/mnt/outputs/stream_b_real/features_cache.joblib'
OUTPUT_DIR = '/sessions/youthful-cool-carson/mnt/outputs/stream_f'

np.random.seed(SEED)


def load_features():
    """Load cached features."""
    print(f"Loading features from {CACHE_PATH}...")
    data = joblib.load(CACHE_PATH)
    X = data['X']
    y = data['y']
    print(f"Loaded: X shape {X.shape}, y shape {y.shape}")
    print(f"y range: [{y.min():.4f}, {y.max():.4f}]")
    return X, y


def evaluate_mlp(X, y, params, seed=SEED):
    """
    Evaluate MLP with given hyperparameters using RepeatedKFold.
    Returns: (mean_r2, std_r2, mean_mae, mean_rmse, fold_scores)
    """
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=seed)
    n_folds = 15

    r2_scores = []
    mae_scores = []
    rmse_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Pipeline: StandardScaler -> MLP
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                alpha=params['alpha'],
                learning_rate=params['learning_rate'],
                learning_rate_init=params['learning_rate_init'],
                max_iter=params['max_iter'],
                early_stopping=params['early_stopping'],
                validation_fraction=params.get('validation_fraction', 0.1) if params['early_stopping'] else 0.1,
                batch_size=params['batch_size'],
                solver=params['solver'],
                random_state=seed,
                verbose=0,
            ))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        r2_scores.append(r2)
        mae_scores.append(mae)
        rmse_scores.append(rmse)

    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)
    mean_mae = np.mean(mae_scores)
    mean_rmse = np.mean(rmse_scores)

    return mean_r2, std_r2, mean_mae, mean_rmse, r2_scores


class MLPObjective:
    """Optuna objective for MLP tuning."""

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.trial_count = 0
        self.trials_data = []

    def __call__(self, trial: optuna.Trial) -> float:
        self.trial_count += 1

        # Define search space
        params = {
            'alpha': trial.suggest_float('alpha', 1e-6, 1e-1, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
            'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive', 'invscaling']),
            'max_iter': trial.suggest_int('max_iter', 500, 5000),
            'early_stopping': trial.suggest_categorical('early_stopping', [True, False]),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 'auto']),
            'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs']),
        }

        # Conditional: validation_fraction only if early_stopping=True
        if params['early_stopping']:
            params['validation_fraction'] = trial.suggest_float('validation_fraction', 0.1, 0.3)
        else:
            params['validation_fraction'] = 0.1

        try:
            mean_r2, std_r2, mean_mae, mean_rmse, fold_scores = evaluate_mlp(
                self.X, self.y, params, seed=SEED
            )

            # Log trial
            trial_record = {
                'trial': self.trial_count,
                'mean_r2': float(mean_r2),
                'std_r2': float(std_r2),
                'mean_mae': float(mean_mae),
                'mean_rmse': float(mean_rmse),
                **params,
            }
            self.trials_data.append(trial_record)

            print(
                f"Trial {self.trial_count:3d}: R²={mean_r2:.4f} ± {std_r2:.4f} | "
                f"MAE={mean_mae:.4f} | RMSE={mean_rmse:.4f} | "
                f"solver={params['solver']} alpha={params['alpha']:.2e}"
            )

            return mean_r2

        except Exception as e:
            print(f"Trial {self.trial_count} failed: {e}")
            return -1.0


def main():
    parser = argparse.ArgumentParser(description="Optuna MLP Tuning")
    parser.add_argument('--n-trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--timeout', type=int, default=600, help='Timeout in seconds (10 min default)')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("STREAM F: OPTUNA BAYESIAN MLP HYPERPARAMETER TUNING")
    print("=" * 90)
    print(f"Seed: {args.seed}")
    print(f"Max trials: {args.n_trials}")
    print(f"Timeout: {args.timeout} seconds")
    print()

    # Load data
    X, y = load_features()

    # Create objective
    objective = MLPObjective(X, y)

    # Create study with TPE sampler (Bayesian)
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        direction='maximize',
    )

    print(f"Starting optimization with TPE sampler (Bayesian)...")
    print()

    start_time = time.time()
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )
    elapsed = time.time() - start_time

    print()
    print("=" * 90)
    print(f"Optimization completed in {elapsed:.1f} seconds ({elapsed/60:.1f} min)")
    print(f"Total trials: {len(study.trials)}")
    print()

    # Best trial
    best_trial = study.best_trial
    best_r2 = best_trial.value
    best_params = best_trial.params

    print("BEST TRIAL:")
    print(f"  Trial #{best_trial.number}: R² = {best_r2:.4f}")
    print("  Parameters:")
    for key, val in sorted(best_params.items()):
        if isinstance(val, float):
            print(f"    {key}: {val:.6f}")
        else:
            print(f"    {key}: {val}")
    print()

    # Re-evaluate best on full CV for full reporting
    print("Re-evaluating best config on full RepeatedKFold...")
    mean_r2, std_r2, mean_mae, mean_rmse, fold_scores = evaluate_mlp(
        X, y, best_params, seed=args.seed
    )

    print(f"  Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"  Mean MAE: {mean_mae:.4f}")
    print(f"  Mean RMSE: {mean_rmse:.4f}")
    print(f"  Per-fold R² scores: {[f'{s:.4f}' for s in fold_scores]}")
    print()

    # Save best params
    best_params_file = output_dir / 'best_mlp_params.json'
    best_params_dict = {
        'best_trial_number': int(best_trial.number),
        'best_r2': float(mean_r2),
        'best_r2_std': float(std_r2),
        'best_mae': float(mean_mae),
        'best_rmse': float(mean_rmse),
        'fold_scores': [float(s) for s in fold_scores],
        'parameters': {
            k: (float(v) if isinstance(v, (int, float, np.number)) else v)
            for k, v in best_params.items()
        },
        'timestamp': datetime.now().isoformat(),
    }

    with open(best_params_file, 'w') as f:
        json.dump(best_params_dict, f, indent=2)

    print(f"Saved best params to: {best_params_file}")

    # Save all trials
    trials_df = pd.DataFrame(objective.trials_data)
    trials_csv = output_dir / 'tuning_trials.csv'
    trials_df.to_csv(trials_csv, index=False)
    print(f"Saved all trials to: {trials_csv}")
    print()

    # Plot optimization history
    print("Generating optimization history plot...")
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization import plot_optimization_history

        fig = optuna.visualization.plot_optimization_history(study).to_plotly_figure()
        fig.write_html(str(output_dir / 'optuna_history.html'))

        # Also save as PDF via matplotlib
        fig_mpl = plt.figure(figsize=(12, 6))
        ax = fig_mpl.add_subplot(111)

        trials_list = study.trials
        trial_nums = [t.number for t in trials_list]
        trial_r2s = [t.value if t.value is not None else -1.0 for t in trials_list]
        best_r2s = [max(trial_r2s[:i+1]) for i in range(len(trial_r2s))]

        ax.plot(trial_nums, trial_r2s, 'o-', alpha=0.5, label='Trial R²')
        ax.plot(trial_nums, best_r2s, 's-', linewidth=2, label='Best R² (cumulative)')
        ax.axhline(y=best_r2, color='r', linestyle='--', label=f'Final best: {best_r2:.4f}')
        ax.axhline(y=0.6631, color='orange', linestyle='--', label='Paper target: 0.6631')
        ax.set_xlabel('Trial')
        ax.set_ylabel('R²')
        ax.set_title('Optuna Optimization History: MLP Tuning')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        pdf_path = output_dir / 'optuna_history.pdf'
        fig_mpl.savefig(pdf_path, dpi=150)
        print(f"Saved optimization history PDF to: {pdf_path}")
        plt.close(fig_mpl)

    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Best R²: {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"Best MAE: {mean_mae:.4f}")
    print(f"Best RMSE: {mean_rmse:.4f}")
    print(f"Paper's target MLP R²: 0.6631")
    print(f"Gap: {0.6631 - mean_r2:.4f} (negative = exceeded target)")
    print()
    print(f"Top hyperparameters:")
    print(f"  solver: {best_params['solver']}")
    print(f"  alpha (L2): {best_params['alpha']:.2e}")
    print(f"  learning_rate_init: {best_params['learning_rate_init']:.2e}")
    print(f"  learning_rate: {best_params['learning_rate']}")
    print(f"  max_iter: {best_params['max_iter']}")
    print(f"  early_stopping: {best_params['early_stopping']}")
    print(f"  batch_size: {best_params['batch_size']}")
    print("=" * 90)


if __name__ == '__main__':
    main()
