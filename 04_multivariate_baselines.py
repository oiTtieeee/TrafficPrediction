"""
================================================================================
Section 8: Multivariate Tabular Baselines (All 207 Sensors)
================================================================================
This script:
1. Ingests the METR-LA dataset.
2. Extracts multivariate flattened lag features across all 207 sensors:
   Input shape: (N_samples, 12 * 207 = 2,484) -> Output: (N_samples, 207).
3. Partitions temporally into 70% Train, 15% Validation, 15% Test splits.
4. Fits and evaluates Multivariate Linear Regression.
5. Fits and evaluates Multivariate Random Forest Regressor.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.config import RAW_DIR, SEQ_LENGTH
from src.data_loader import load_metr_la
from src.preprocessing import create_multivariate_sequences
from src.models.baselines import train_linear_regression, train_random_forest
from src.evaluate import calculate_metrics

def main():
    print("=" * 70)
    print("SECTION 8: MULTIVARIATE TABULAR BASELINES ACROSS ALL 207 SENSORS")
    print("=" * 70)

    # 1. Load data
    df = load_metr_la(raw_dir=RAW_DIR)
    values = df.values
    print(f"Loaded speed matrix: {values.shape}")

    # 2. Build multivariate flattened sequences (12 timesteps * 207 sensors = 2,484 features)
    print(f"\n[1] Extracting multivariate lag features (SEQ_LENGTH = {SEQ_LENGTH})...")
    X, y = create_multivariate_sequences(values, seq_length=SEQ_LENGTH)
    print(f"    Feature matrix X: {X.shape}")
    print(f"    Target matrix y:   {y.shape}")

    # 3. Chronological Time-Series Split (70% Train, 15% Val, 15% Test)
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val     = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test   = X[val_end:], y[val_end:]

    print(f"\n[2] Chronological Split: Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    # 4. Multivariate Linear Regression
    print("\n[3] Fitting Multivariate Linear Regression...")
    lr = train_linear_regression(X_train, y_train)
    pred_lr = lr.predict(X_test)
    mae_lr, rmse_lr = calculate_metrics(y_test, pred_lr)
    print(f"    Multivariate Linear Regression -> MAE: {mae_lr:.3f}, RMSE: {rmse_lr:.3f}")

    # 5. Multivariate Random Forest
    print("\n[4] Fitting Multivariate Random Forest Regressor (n_estimators=20, max_depth=10)...")
    rf = train_random_forest(X_train, y_train, n_estimators=20, max_depth=10, n_jobs=-1, random_state=42)
    pred_rf = rf.predict(X_test)
    mae_rf, rmse_rf = calculate_metrics(y_test, pred_rf)
    print(f"    Multivariate Random Forest    -> MAE: {mae_rf:.3f}, RMSE: {rmse_rf:.3f}")

    print("\nDone! Section 8 completed successfully.")

if __name__ == "__main__":
    main()
