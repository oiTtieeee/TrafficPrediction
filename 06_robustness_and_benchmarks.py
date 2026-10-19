"""
================================================================================
Section 12-14: Data-Constrained Robustness, Comparative Benchmarks & Export
================================================================================
This script:
1. Evaluates model robustness under data scarcity: 100%, 50%, and 20% training history.
2. Trains and evaluates TrafficGCN-Delta across all three regimes.
3. Trains and evaluates classical baselines (LR & RF) on identical data subsets.
4. Synthesizes the benchmark comparison table and exports to outputs/final_results.csv.
5. Saves model state checkpoints and prediction arrays.
"""
import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from src.config import RAW_DIR, CACHE_DIR, CHECKPOINT_DIR, OUTPUTS_DIR, SEQ_LENGTH, DEVICE
from src.data_loader import load_metr_la, download_graph_files
from src.graph_utils import build_adjacency, adjacency_to_edge_tensors
from src.preprocessing import clean_and_normalize, create_graph_sequences, create_multivariate_sequences, get_data_subset
from src.models.baselines import train_linear_regression, train_random_forest
from src.train import train_gcn_delta
from src.evaluate import evaluate_gcn_delta, calculate_metrics, append_result_row

def main():
    print("=" * 70)
    print("SECTION 12: DATA-CONSTRAINED ROBUSTNESS EXPERIMENTS (100%, 50%, 20%)")
    print("=" * 70)

    # 1. Prepare Data & Topology
    df = load_metr_la(raw_dir=RAW_DIR)
    values = df.values

    # Graph tensors
    graph_tensors_path = CACHE_DIR / "graph_tensors.pt"
    if graph_tensors_path.exists():
        graph_cache = torch.load(graph_tensors_path, weights_only=False)
        edge_index = graph_cache["edge_index"]
        edge_weight = graph_cache["edge_weight"]
    else:
        paths = download_graph_files(raw_dir=RAW_DIR)
        with open(paths["graph_sensor_ids.txt"]) as f:
            sensor_ids = f.read().strip().split(',')
        dist_df = pd.read_csv(paths["distances_la_2012.csv"], dtype={'from': str, 'to': str})
        adj_mx = build_adjacency(sensor_ids, dist_df)
        edge_index, edge_weight = adjacency_to_edge_tensors(adj_mx)

    # Graph sequences
    values_norm, mean, std = clean_and_normalize(df, train_ratio=0.70)
    X_graph, y_graph = create_graph_sequences(values_norm, seq_length=SEQ_LENGTH)

    n = len(X_graph)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train_g, y_train_g = X_graph[:train_end], y_graph[:train_end]
    X_val_g, y_val_g     = X_graph[train_end:val_end], y_graph[train_end:val_end]
    X_test_g, y_test_g   = X_graph[val_end:], y_graph[val_end:]

    # Multivariate flattened sequences (for LR and RF)
    X_multi, y_multi = create_multivariate_sequences(values, seq_length=SEQ_LENGTH)
    X_train_m, y_train_m = X_multi[:train_end], y_multi[:train_end]
    X_test_m, y_test_m   = X_multi[val_end:], y_multi[val_end:]

    results = {}
    last_predictions = None

    # 2. Train and Evaluate TrafficGCN-Delta across regimes
    regimes = [(1.0, '100%'), (0.5, '50%'), (0.2, '20%')]
    for frac, label in regimes:
        print(f"\n>>> [GCN] Training TrafficGCN-Delta at {label} data (frac={frac})...")
        X_sub, y_sub = get_data_subset(X_train_g, y_train_g, frac)
        print(f"    Training samples available: {X_sub.shape[0]}")

        model_frac = train_gcn_delta(
            X_sub, y_sub, X_val_g, y_val_g,
            edge_index, edge_weight,
            epochs=30, batch_size=64, lr=0.001, device=DEVICE
        )

        mae, rmse, preds = evaluate_gcn_delta(
            model_frac, X_test_g, y_test_g,
            edge_index, edge_weight, mean, std, device=DEVICE
        )
        print(f"    GCN @ {label} -> MAE: {mae:.3f}, RMSE: {rmse:.3f}")
        results[('GCN', label)] = (mae, rmse)
        last_predictions = preds

        # Checkpoint model
        ckpt_path = CHECKPOINT_DIR / f"gcn_delta_{label.replace('%', 'pct')}.pt"
        torch.save(model_frac.state_dict(), ckpt_path)
        append_result_row("GCN", label, mae, rmse)

    # 3. Train and Evaluate Classical Baselines across identical regimes
    for frac, label in regimes:
        print(f"\n>>> [Baselines] Training LR & RF at {label} data...")
        X_sub_m, y_sub_m = get_data_subset(X_train_m, y_train_m, frac)

        # Linear Regression
        lr = train_linear_regression(X_sub_m, y_sub_m)
        pred_lr = lr.predict(X_test_m)
        mae_lr, rmse_lr = calculate_metrics(y_test_m, pred_lr)
        results[('LR', label)] = (mae_lr, rmse_lr)
        append_result_row("LR", label, mae_lr, rmse_lr)
        print(f"    LR  @ {label} -> MAE: {mae_lr:.3f}, RMSE: {rmse_lr:.3f}")

        # Random Forest
        rf = train_random_forest(X_sub_m, y_sub_m, n_estimators=25, max_depth=10, n_jobs=-1, random_state=42)
        pred_rf = rf.predict(X_test_m)
        mae_rf, rmse_rf = calculate_metrics(y_test_m, pred_rf)
        results[('RF', label)] = (mae_rf, rmse_rf)
        append_result_row("RF", label, mae_rf, rmse_rf)
        print(f"    RF  @ {label} -> MAE: {mae_rf:.3f}, RMSE: {rmse_rf:.3f}")

    # 4. Synthesize Final Comparison Table
    print("\n" + "=" * 70)
    print("SECTION 13: COMPARATIVE BENCHMARK SYNTHESIS & EXPORT")
    print("=" * 70)

    rows = []
    for model_name in ['LR', 'RF', 'GCN']:
        row = {'Model': model_name}
        for _, label in regimes:
            mae, rmse = results[(model_name, label)]
            row[f'MAE_{label}'] = round(mae, 3)
            row[f'RMSE_{label}'] = round(rmse, 3)
        rows.append(row)

    results_df = pd.DataFrame(rows)
    print("\nBenchmark Summary Table:")
    print(results_df.to_string(index=False))

    final_csv_path = OUTPUTS_DIR / "final_results.csv"
    results_df.to_csv(final_csv_path, index=False)
    print(f"\nSaved final summary table to: {final_csv_path}")

    # Save predictions array
    if last_predictions is not None:
        pred_path = OUTPUTS_DIR / "predictions.npy"
        np.save(pred_path, last_predictions)
        print(f"Saved final predictions array to: {pred_path}")

    print("\nDone! Full benchmark evaluation completed successfully.")

if __name__ == "__main__":
    main()
