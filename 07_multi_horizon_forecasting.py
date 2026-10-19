"""
================================================================================
Experiment: Multi-Horizon Traffic Speed Forecasting (5, 15, 30, 60 minutes)
================================================================================
This script:
1. Loads the METR-LA dataset and computes train-set normalization statistics.
2. Constructs multi-horizon prediction targets:
   - 5-min ahead  (h = 1 step)
   - 15-min ahead (h = 3 steps)
   - 30-min ahead (h = 6 steps)
   - 60-min ahead (h = 12 steps)
3. Trains and evaluates:
   - TrafficGCN-Delta (Proposed residual velocity momentum formulation)
   - Direct TrafficGCN (Direct speed prediction)
   - Linear Regression (Multivariate)
   - Random Forest (Multivariate)
   - Persistence Baseline
4. Logs results to outputs/multi_horizon_results.csv.
5. Generates publication-ready figure: paper/figures/fig6_multi_horizon_forecasting.png.
"""
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.config import RAW_DIR, CACHE_DIR, OUTPUTS_DIR, SEQ_LENGTH
from src.data_loader import load_metr_la, download_graph_files
from src.graph_utils import build_adjacency
from src.preprocessing import clean_and_normalize, create_multivariate_sequences
from src.models.gcn import TrafficGCN
from src.models.baselines import train_linear_regression, train_random_forest
from src.evaluate import calculate_metrics

def compute_normalized_laplacian(adj_mx):
    adj = adj_mx.copy()
    np.fill_diagonal(adj, 1.0)
    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[deg == 0] = 0.0
    d_mat = np.diag(deg_inv_sqrt)
    adj_norm = d_mat @ adj @ d_mat
    return torch.tensor(adj_norm, dtype=torch.float32)

def create_multi_horizon_sequences(data, seq_length=12, horizons=[1, 3, 6, 12]):
    max_h = max(horizons)
    X = []
    y_dict = {h: [] for h in horizons}
    for i in range(len(data) - seq_length - max_h + 1):
        X.append(data[i:i + seq_length].T)
        for h in horizons:
            y_dict[h].append(data[i + seq_length + h - 1])
    X = np.array(X, dtype=np.float32)
    for h in horizons:
        y_dict[h] = np.array(y_dict[h], dtype=np.float32)
    return X, y_dict

def train_gcn_horizon(X_train, y_train, X_val, y_val, adj_norm, is_delta=True, epochs=20, batch_size=64, lr=0.001, device='cpu'):
    model = TrafficGCN(in_channels=X_train.shape[2], hidden_channels=64, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    n_train = len(X_train)
    adj_norm = adj_norm.to(device)

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)
        total_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            batch_x = torch.tensor(X_train[idx], dtype=torch.float32, device=device)
            batch_last = batch_x[:, :, -1]
            batch_y = torch.tensor(y_train[idx], dtype=torch.float32, device=device)

            if is_delta:
                target = batch_y - batch_last
            else:
                target = batch_y

            optimizer.zero_grad()
            pred = model(batch_x, adj_norm=adj_norm)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

    return model

def evaluate_gcn_horizon(model, X_test, y_test, adj_norm, mean, std, is_delta=True, batch_size=64, device='cpu'):
    model.eval()
    preds_all = []
    adj_norm = adj_norm.to(device)
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_x = torch.tensor(X_test[i:i + batch_size], dtype=torch.float32, device=device)
            batch_last = batch_x[:, :, -1]
            pred = model(batch_x, adj_norm=adj_norm)
            if is_delta:
                pred = batch_last + pred
            preds_all.append(pred.cpu().numpy())

    preds = np.concatenate(preds_all) * std + mean
    actuals = y_test * std + mean
    mae, rmse = calculate_metrics(actuals, preds)
    return mae, rmse

def main():
    print("=" * 75)
    print("EXPERIMENT: MULTI-HORIZON TRAFFIC SPEED FORECASTING (5, 15, 30, 60 MIN)")
    print("=" * 75)

    # 1. Load Data
    df = load_metr_la(raw_dir=RAW_DIR)
    print(f"Loaded dataset: {df.shape}")

    # 2. Build Adjacency Matrix
    paths = download_graph_files(raw_dir=RAW_DIR)
    with open(paths["graph_sensor_ids.txt"]) as f:
        sensor_ids = f.read().strip().split(',')
    dist_df = pd.read_csv(paths["distances_la_2012.csv"], dtype={'from': str, 'to': str})
    adj_mx = build_adjacency(sensor_ids, dist_df)
    adj_norm = compute_normalized_laplacian(adj_mx)
    print(f"Normalized Laplacian computed: {adj_norm.shape}")

    # 3. Clean and Normalize
    values_norm, mean, std = clean_and_normalize(df, train_ratio=0.70)
    print(f"Normalization statistics: mean={mean:.4f}, std={std:.4f}")

    # 4. Multi-Horizon Sequences
    horizons = [1, 3, 6, 12]  # 5 min, 15 min, 30 min, 60 min
    horizon_names = {1: "5-min (h=1)", 3: "15-min (h=3)", 6: "30-min (h=6)", 12: "60-min (h=12)"}
    X_graph, y_dict = create_multi_horizon_sequences(values_norm, seq_length=SEQ_LENGTH, horizons=horizons)

    n = len(X_graph)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train_g = X_graph[:train_end]
    X_val_g   = X_graph[train_end:val_end]
    X_test_g  = X_graph[val_end:]

    print(f"Total samples: {n} (Train: {train_end}, Val: {val_end - train_end}, Test: {n - val_end})")

    results_table = []

    # 5. Evaluate Multi-Horizon for each model
    for h in horizons:
        h_name = horizon_names[h]
        print(f"\n" + "-" * 60)
        print(f"EVALUATING HORIZON: {h_name}")
        print("-" * 60)

        y_train = y_dict[h][:train_end]
        y_val   = y_dict[h][train_end:val_end]
        y_test  = y_dict[h][val_end:]

        # A. Persistence Baseline: y_hat = x[:,-1]
        last_val_test = X_test_g[:, :, -1] * std + mean
        actual_test = y_test * std + mean
        mae_persist, rmse_persist = calculate_metrics(actual_test, last_val_test)
        print(f"  [Persistence]      MAE: {mae_persist:.3f} mph | RMSE: {rmse_persist:.3f} mph")
        results_table.append({"Horizon": h_name, "Model": "Persistence", "MAE": mae_persist, "RMSE": rmse_persist})

        # B. Linear Regression (Multivariate)
        X_train_flat = X_train_g.transpose(0, 2, 1).reshape(len(X_train_g), -1)
        X_test_flat  = X_test_g.transpose(0, 2, 1).reshape(len(X_test_g), -1)
        lr = train_linear_regression(X_train_flat[:8000], (y_train[:8000] * std + mean))
        pred_lr = lr.predict(X_test_flat)
        mae_lr, rmse_lr = calculate_metrics(actual_test, pred_lr)
        print(f"  [Linear Reg]       MAE: {mae_lr:.3f} mph | RMSE: {rmse_lr:.3f} mph")
        results_table.append({"Horizon": h_name, "Model": "Linear Regression", "MAE": mae_lr, "RMSE": rmse_lr})

        # C. Direct TrafficGCN (predicting normalized speed directly)
        print(f"  Training Direct TrafficGCN for {h_name}...")
        model_direct = train_gcn_horizon(X_train_g, y_train, X_val_g, y_val, adj_norm, is_delta=False, epochs=15, batch_size=64, lr=0.005)
        mae_dir, rmse_dir = evaluate_gcn_horizon(model_direct, X_test_g, y_test, adj_norm, mean, std, is_delta=False)
        print(f"  [Direct GCN]       MAE: {mae_dir:.3f} mph | RMSE: {rmse_dir:.3f} mph")
        results_table.append({"Horizon": h_name, "Model": "Direct TrafficGCN", "MAE": mae_dir, "RMSE": rmse_dir})

        # D. TrafficGCN-Delta (Proposed residual momentum formulation)
        print(f"  Training TrafficGCN-Delta for {h_name}...")
        model_delta = train_gcn_horizon(X_train_g, y_train, X_val_g, y_val, adj_norm, is_delta=True, epochs=20, batch_size=64, lr=0.001)
        mae_delta, rmse_delta = evaluate_gcn_horizon(model_delta, X_test_g, y_test, adj_norm, mean, std, is_delta=True)
        print(f"  [TrafficGCN-Delta] MAE: {mae_delta:.3f} mph | RMSE: {rmse_delta:.3f} mph")
        results_table.append({"Horizon": h_name, "Model": "TrafficGCN-Delta (Proposed)", "MAE": mae_delta, "RMSE": rmse_delta})

    # Convert to DataFrame
    res_df = pd.DataFrame(results_table)
    out_csv = OUTPUTS_DIR / "multi_horizon_results.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved multi-horizon benchmark results to: {out_csv}")
    print("\n" + str(res_df))

    # 6. Generate Publication Figure
    fig_dir = Path("paper/figures")
    fig_dir.mkdir(exist_ok=True)
    fig_path = fig_dir / "fig6_multi_horizon_forecasting.png"

    plt.figure(figsize=(9, 4.5), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    models = ["Persistence", "Linear Regression", "Direct TrafficGCN", "TrafficGCN-Delta (Proposed)"]
    colors = ['#888888', '#e74c3c', '#e67e22', '#2ecc71']
    markers = ['o', 's', '^', 'D']

    plt.subplot(1, 2, 1)
    for m, c, mk in zip(models, colors, markers):
        sub = res_df[res_df["Model"] == m]
        plt.plot([5, 15, 30, 60], sub["MAE"].values, marker=mk, color=c, label=m, linewidth=2, markersize=7)
    plt.title("(a) Multi-Horizon Mean Absolute Error (MAE)", fontsize=11, fontweight='bold')
    plt.xlabel("Forecasting Horizon (Minutes)", fontsize=10)
    plt.ylabel("MAE (mph)", fontsize=10)
    plt.xticks([5, 15, 30, 60], ['5 min', '15 min', '30 min', '60 min'])
    plt.legend(frameon=True, fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    for m, c, mk in zip(models, colors, markers):
        sub = res_df[res_df["Model"] == m]
        plt.plot([5, 15, 30, 60], sub["RMSE"].values, marker=mk, color=c, label=m, linewidth=2, markersize=7)
    plt.title("(b) Multi-Horizon Root Mean Squared Error (RMSE)", fontsize=11, fontweight='bold')
    plt.xlabel("Forecasting Horizon (Minutes)", fontsize=10)
    plt.ylabel("RMSE (mph)", fontsize=10)
    plt.xticks([5, 15, 30, 60], ['5 min', '15 min', '30 min', '60 min'])
    plt.legend(frameon=True, fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"Generated multi-horizon figure at: {fig_path}")

if __name__ == "__main__":
    main()
