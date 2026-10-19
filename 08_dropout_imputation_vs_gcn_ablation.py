"""
================================================================================
Experiment: Sensor Dropout Imputation vs. GCN Topological Message-Passing Ablation
================================================================================
This script conducts a 2x2 factorial ablation to dissect the relative contributions
of temporal imputation (Forward-Fill vs. Zero-Fill) and spatial graph message-passing
(TrafficGCN-Delta vs. MLP-Delta) under simulated sensor observation loss:
1. Multi-tier random dropouts: 10%, 20%, 30%, 50%
2. Contiguous 2-hour block outage (20% of network sensors)

Evaluates:
- Zero-Fill + MLP-Delta (No Imputation, No Graph)
- Zero-Fill + TrafficGCN-Delta (No Imputation, With Graph)
- Forward-Fill + MLP-Delta (With Imputation, No Graph)
- Forward-Fill + TrafficGCN-Delta (With Imputation, With Graph - Proposed)

Outputs:
- outputs/dropout_ablation_results.csv
- paper/figures/fig7_dropout_imputation_ablation.png
================================================================================
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
import matplotlib.pyplot as plt

from src.config import RAW_DIR, CACHE_DIR, OUTPUTS_DIR, SEQ_LENGTH
from src.data_loader import load_metr_la, download_graph_files
from src.graph_utils import build_adjacency
from src.preprocessing import clean_and_normalize, create_graph_sequences
from src.models.gcn import TrafficGCN
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

def train_model(X_train, y_train, adj_matrix, is_delta=True, epochs=20, batch_size=64, lr=0.001, device='cpu'):
    model = TrafficGCN(in_channels=X_train.shape[2], hidden_channels=64, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    n_train = len(X_train)
    adj_matrix = adj_matrix.to(device)

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            batch_x = torch.tensor(X_train[idx], dtype=torch.float32, device=device)
            batch_last = batch_x[:, :, -1]
            batch_y = torch.tensor(y_train[idx], dtype=torch.float32, device=device)

            target = batch_y - batch_last if is_delta else batch_y

            optimizer.zero_grad()
            pred = model(batch_x, adj_norm=adj_matrix)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

    return model

def apply_random_dropout(X, dropout_rate, seed=42):
    """
    Simulates random sensor packet dropouts.
    Returns:
    - X_zero: missing entries set to 0.0 (dataset normalized mean)
    - X_ffill: missing entries forward-filled along the temporal dimension
    """
    np.random.seed(seed)
    mask = np.random.rand(*X.shape) < dropout_rate # True where dropped

    # Zero fill
    X_zero = X.copy()
    X_zero[mask] = 0.0

    # Forward fill
    X_ffill = X.copy()
    B, N, T = X.shape
    for b in range(B):
        for n in range(N):
            last_val = 0.0
            for t in range(T):
                if mask[b, n, t]:
                    X_ffill[b, n, t] = last_val
                else:
                    last_val = X_ffill[b, n, t]

    return X_zero, X_ffill

def apply_block_outage(X, outage_sensor_frac=0.20, seed=42):
    """
    Simulates contiguous block outage where a fraction of sensors completely fail
    across the entire lookback window.
    """
    np.random.seed(seed)
    B, N, T = X.shape
    num_failed = int(N * outage_sensor_frac)
    failed_indices = np.random.choice(N, size=num_failed, replace=False)

    X_zero = X.copy()
    X_zero[:, failed_indices, :] = 0.0

    # For block outage, forward-fill has no lookback to fill from within the window,
    # so forward-fill defaults to 0.0 (mean), but let's test if prior history was present
    X_ffill = X_zero.copy()

    return X_zero, X_ffill

def evaluate_model(model, X_test, y_test, adj_matrix, mean, std, is_delta=True, batch_size=64, device='cpu'):
    model.eval()
    preds_all = []
    adj_matrix = adj_matrix.to(device)

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            bx = torch.tensor(X_test[i:i + batch_size], dtype=torch.float32, device=device)
            last_x = bx[:, :, -1].cpu().numpy()
            pred_delta = model(bx, adj_norm=adj_matrix).cpu().numpy()
            if is_delta:
                pred = last_x + pred_delta
            else:
                pred = pred_delta
            preds_all.append(pred)

    preds_norm = np.concatenate(preds_all, axis=0)
    preds_real = preds_norm * std + mean
    y_real = y_test * std + mean

    mae = np.mean(np.abs(preds_real - y_real))
    rmse = np.sqrt(np.mean((preds_real - y_real) ** 2))
    return mae, rmse

def main():
    print("=" * 80)
    print("SENSOR DROPOUT IMPUTATION-VS-GCN ABLATION EXPERIMENT")
    print("=" * 80)

    # 1. Load Data
    print("\n[1] Loading METR-LA dataset...")
    df = load_metr_la(raw_dir=RAW_DIR)
    values_norm, mean, std = clean_and_normalize(df, train_ratio=0.70)
    X_graph, y_graph = create_graph_sequences(values_norm, seq_length=SEQ_LENGTH)

    n = len(X_graph)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X_graph[:train_end], y_graph[:train_end]
    X_val, y_val = X_graph[train_end:val_end], y_graph[train_end:val_end]
    X_test, y_test = X_graph[val_end:], y_graph[val_end:]

    print(f"    Train samples: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 2. Build Adjacency and Identity Matrices
    print("\n[2] Building Graph Laplacians (Road Graph & Identity)...")
    paths = download_graph_files(raw_dir=RAW_DIR)
    with open(paths["graph_sensor_ids.txt"]) as f:
        sensor_ids = f.read().strip().split(',')
    dist_df = pd.read_csv(paths["distances_la_2012.csv"], dtype={'from': str, 'to': str})
    adj_mx = build_adjacency(sensor_ids, dist_df)

    adj_norm = compute_normalized_laplacian(adj_mx)
    identity_adj = torch.eye(adj_norm.shape[0], dtype=torch.float32)

    # 3. Train Models
    print("\n[3] Training TrafficGCN-Delta (With Graph) for 20 epochs...")
    t0 = time.time()
    gcn_model = train_model(X_train, y_train, adj_norm, is_delta=True, epochs=20, batch_size=64)
    print(f"    TrafficGCN-Delta trained in {time.time() - t0:.2f}s")

    print("\n[4] Training MLP-Delta (No Graph, Identity Laplacian) for 20 epochs...")
    t0 = time.time()
    mlp_model = train_model(X_train, y_train, identity_adj, is_delta=True, epochs=20, batch_size=64)
    print(f"    MLP-Delta trained in {time.time() - t0:.2f}s")

    # 4. Evaluate under Clean Conditions (0% Dropout)
    print("\n[5] Evaluating under Clean Conditions (0% Dropout)...")
    mae_clean_gcn, rmse_clean_gcn = evaluate_model(gcn_model, X_test, y_test, adj_norm, mean, std)
    mae_clean_mlp, rmse_clean_mlp = evaluate_model(mlp_model, X_test, y_test, identity_adj, mean, std)
    print(f"    Clean TrafficGCN-Delta -> MAE: {mae_clean_gcn:.3f}, RMSE: {rmse_clean_gcn:.3f}")
    print(f"    Clean MLP-Delta        -> MAE: {mae_clean_mlp:.3f}, RMSE: {rmse_clean_mlp:.3f}")

    results = []

    # Record clean baseline
    for model_name, graph_mode in [("MLP-Delta (No Graph)", "No Graph"), ("TrafficGCN-Delta (Proposed)", "With Graph")]:
        for imp_name in ["Zero-Fill (No Imputation)", "Forward-Fill"]:
            mae = mae_clean_mlp if "No Graph" in model_name else mae_clean_gcn
            rmse = rmse_clean_mlp if "No Graph" in model_name else rmse_clean_gcn
            results.append({
                "Condition": "Clean (0%)",
                "Dropout_Rate": 0.0,
                "Imputation": imp_name,
                "Model": model_name,
                "Graph": graph_mode,
                "MAE": mae,
                "RMSE": rmse
            })

    # 5. Evaluate Multi-Tier Random Dropouts
    dropouts = [0.10, 0.20, 0.30, 0.50]
    print("\n[6] Evaluating Multi-Tier Random Dropouts (10%, 20%, 30%, 50%)...")

    for p in dropouts:
        pct_label = f"{int(p*100)}% Dropout"
        print(f"\n--- Testing {pct_label} ---")
        X_zero, X_ffill = apply_random_dropout(X_test, dropout_rate=p, seed=42)

        # 1. Zero-Fill + MLP-Delta
        mae_z_mlp, rmse_z_mlp = evaluate_model(mlp_model, X_zero, y_test, identity_adj, mean, std)
        results.append({
            "Condition": pct_label,
            "Dropout_Rate": p,
            "Imputation": "Zero-Fill (No Imputation)",
            "Model": "MLP-Delta (No Graph)",
            "Graph": "No Graph",
            "MAE": mae_z_mlp,
            "RMSE": rmse_z_mlp
        })

        # 2. Zero-Fill + TrafficGCN-Delta
        mae_z_gcn, rmse_z_gcn = evaluate_model(gcn_model, X_zero, y_test, adj_norm, mean, std)
        results.append({
            "Condition": pct_label,
            "Dropout_Rate": p,
            "Imputation": "Zero-Fill (No Imputation)",
            "Model": "TrafficGCN-Delta (Proposed)",
            "Graph": "With Graph",
            "MAE": mae_z_gcn,
            "RMSE": rmse_z_gcn
        })

        # 3. Forward-Fill + MLP-Delta
        mae_f_mlp, rmse_f_mlp = evaluate_model(mlp_model, X_ffill, y_test, identity_adj, mean, std)
        results.append({
            "Condition": pct_label,
            "Dropout_Rate": p,
            "Imputation": "Forward-Fill",
            "Model": "MLP-Delta (No Graph)",
            "Graph": "No Graph",
            "MAE": mae_f_mlp,
            "RMSE": rmse_f_mlp
        })

        # 4. Forward-Fill + TrafficGCN-Delta
        mae_f_gcn, rmse_f_gcn = evaluate_model(gcn_model, X_ffill, y_test, adj_norm, mean, std)
        results.append({
            "Condition": pct_label,
            "Dropout_Rate": p,
            "Imputation": "Forward-Fill",
            "Model": "TrafficGCN-Delta (Proposed)",
            "Graph": "With Graph",
            "MAE": mae_f_gcn,
            "RMSE": rmse_f_gcn
        })

        print(f"    Zero-Fill    + MLP-Delta   : MAE = {mae_z_mlp:.3f}, RMSE = {rmse_z_mlp:.3f}")
        print(f"    Zero-Fill    + TrafficGCN  : MAE = {mae_z_gcn:.3f}, RMSE = {rmse_z_gcn:.3f}")
        print(f"    Forward-Fill + MLP-Delta   : MAE = {mae_f_mlp:.3f}, RMSE = {rmse_f_mlp:.3f}")
        print(f"    Forward-Fill + TrafficGCN  : MAE = {mae_f_gcn:.3f}, RMSE = {rmse_f_gcn:.3f}")

    # 6. Evaluate 2-Hour Block Outage
    print("\n[7] Evaluating 2-Hour Contiguous Block Outage (20% of network sensors)...")
    X_block_zero, X_block_ffill = apply_block_outage(X_test, outage_sensor_frac=0.20, seed=42)

    mae_bz_mlp, rmse_bz_mlp = evaluate_model(mlp_model, X_block_zero, y_test, identity_adj, mean, std)
    mae_bz_gcn, rmse_bz_gcn = evaluate_model(gcn_model, X_block_zero, y_test, adj_norm, mean, std)
    mae_bf_mlp, rmse_bf_mlp = evaluate_model(mlp_model, X_block_ffill, y_test, identity_adj, mean, std)
    mae_bf_gcn, rmse_bf_gcn = evaluate_model(gcn_model, X_block_ffill, y_test, adj_norm, mean, std)

    results.append({
        "Condition": "2-Hour Block Outage",
        "Dropout_Rate": 0.20,
        "Imputation": "Zero-Fill (No Imputation)",
        "Model": "MLP-Delta (No Graph)",
        "Graph": "No Graph",
        "MAE": mae_bz_mlp,
        "RMSE": rmse_bz_mlp
    })
    results.append({
        "Condition": "2-Hour Block Outage",
        "Dropout_Rate": 0.20,
        "Imputation": "Zero-Fill (No Imputation)",
        "Model": "TrafficGCN-Delta (Proposed)",
        "Graph": "With Graph",
        "MAE": mae_bz_gcn,
        "RMSE": rmse_bz_gcn
    })
    results.append({
        "Condition": "2-Hour Block Outage",
        "Dropout_Rate": 0.20,
        "Imputation": "Forward-Fill",
        "Model": "MLP-Delta (No Graph)",
        "Graph": "No Graph",
        "MAE": mae_bf_mlp,
        "RMSE": rmse_bf_mlp
    })
    results.append({
        "Condition": "2-Hour Block Outage",
        "Dropout_Rate": 0.20,
        "Imputation": "Forward-Fill",
        "Model": "TrafficGCN-Delta (Proposed)",
        "Graph": "With Graph",
        "MAE": mae_bf_gcn,
        "RMSE": rmse_bf_gcn
    })

    print(f"    Block Outage | Zero-Fill    + MLP-Delta   : MAE = {mae_bz_mlp:.3f}")
    print(f"    Block Outage | Zero-Fill    + TrafficGCN  : MAE = {mae_bz_gcn:.3f}")
    print(f"    Block Outage | Forward-Fill + MLP-Delta   : MAE = {mae_bf_mlp:.3f}")
    print(f"    Block Outage | Forward-Fill + TrafficGCN  : MAE = {mae_bf_gcn:.3f}")

    # 7. Save to CSV
    results_df = pd.DataFrame(results)
    csv_path = OUTPUTS_DIR / "dropout_ablation_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n[8] Saved results to: {csv_path}")

    # 8. Generate Publication Figure
    print("\n[9] Generating publication figure...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Plot 1: Multi-Tier Random Dropouts
    plot_df = results_df[results_df["Condition"].str.contains("Dropout|Clean")].copy()
    plot_df = plot_df.drop_duplicates(subset=["Condition", "Imputation", "Model"])

    conditions = ["Clean (0%)", "10% Dropout", "20% Dropout", "30% Dropout", "50% Dropout"]
    rates = [0, 10, 20, 30, 50]

    series_data = {
        "Zero-Fill + MLP (No Graph)": ("Zero-Fill (No Imputation)", "MLP-Delta (No Graph)", "#d9534f", "s--"),
        "Zero-Fill + GCN (Spatial Only)": ("Zero-Fill (No Imputation)", "TrafficGCN-Delta (Proposed)", "#f0ad4e", "^--"),
        "Forward-Fill + MLP (Temporal Only)": ("Forward-Fill", "MLP-Delta (No Graph)", "#5bc0de", "o-"),
        "Forward-Fill + GCN (Proposed)": ("Forward-Fill", "TrafficGCN-Delta (Proposed)", "#2b8a3e", "D-")
    }

    for label, (imp, mod, color, fmt) in series_data.items():
        vals = []
        for c in conditions:
            sub = plot_df[(plot_df["Condition"] == c) & (plot_df["Imputation"] == imp) & (plot_df["Model"] == mod)]
            vals.append(sub["MAE"].values[0] if len(sub) > 0 else np.nan)
        ax1.plot(rates, vals, fmt, color=color, linewidth=2, markersize=7, label=label)

    ax1.set_title("(a) Multi-Tier Random Sensor Dropouts", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Sensor Dropout Rate (%)", fontsize=11)
    ax1.set_ylabel("Speed MAE (mph)", fontsize=11)
    ax1.set_xticks(rates)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Plot 2: Block Outage Comparison
    block_sub = results_df[results_df["Condition"] == "2-Hour Block Outage"].copy()
    bar_labels = [
        "Zero-Fill\n+ MLP",
        "Zero-Fill\n+ GCN",
        "Forward-Fill\n+ MLP",
        "Forward-Fill\n+ GCN\n(Proposed)"
    ]
    bar_vals = [
        block_sub[(block_sub["Imputation"] == "Zero-Fill (No Imputation)") & (block_sub["Model"] == "MLP-Delta (No Graph)")]["MAE"].values[0],
        block_sub[(block_sub["Imputation"] == "Zero-Fill (No Imputation)") & (block_sub["Model"] == "TrafficGCN-Delta (Proposed)")]["MAE"].values[0],
        block_sub[(block_sub["Imputation"] == "Forward-Fill") & (block_sub["Model"] == "MLP-Delta (No Graph)")]["MAE"].values[0],
        block_sub[(block_sub["Imputation"] == "Forward-Fill") & (block_sub["Model"] == "TrafficGCN-Delta (Proposed)")]["MAE"].values[0]
    ]
    bar_colors = ["#d9534f", "#f0ad4e", "#5bc0de", "#2b8a3e"]

    bars = ax2.bar(bar_labels, bar_vals, color=bar_colors, width=0.55, edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, bar_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.08, f"{val:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_title("(b) 2-Hour Contiguous Block Outage (20% Sensors)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Speed MAE (mph)", fontsize=11)
    ax2.grid(axis='y', linestyle="--", alpha=0.5)
    ax2.set_ylim(0, max(bar_vals) * 1.2)

    plt.tight_layout()
    fig_path = PROJECT_ROOT / "paper" / "figures" / "fig7_dropout_imputation_ablation.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved figure to: {fig_path}")

    print("\n" + "=" * 80)
    print("DROPOUT IMPUTATION-VS-GCN ABLATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
