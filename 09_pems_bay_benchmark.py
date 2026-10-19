"""
================================================================================
Script 09: PEMS-BAY Cross-Dataset Benchmark (High-Performance Vectorized)
================================================================================
Validates TrafficGCN-Delta on the PEMS-BAY dataset (325 sensors, San Francisco
Bay Area) across three data-availability regimes: 100%, 50%, and 20%.

Outputs:
  outputs/pems_bay_results.csv   -- Full benchmark table
  models/checkpoints/pems_bay_*  -- Model checkpoints

Usage:
  python scripts/09_pems_bay_benchmark.py
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
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from src.config import RAW_DIR, CHECKPOINT_DIR, OUTPUTS_DIR, SEQ_LENGTH
from src.data_loader_pems import load_pems_bay, build_pems_bay_adjacency
from src.preprocessing import clean_and_normalize, create_graph_sequences, create_multivariate_sequences, get_data_subset
from src.models.gcn import TrafficGCN
from src.evaluate import calculate_metrics

# ── Constants ──────────────────────────────────────────────────────────────────
PEMS_RAW_DIR = RAW_DIR
RESULTS_PATH = OUTPUTS_DIR / "pems_bay_results.csv"
REGIMES      = [(1.0, "100%"), (0.5, "50%"), (0.2, "20%")]
N_SEEDS      = 3                # 3 independent seeds for mean ± std
EPOCHS       = 10               # 10 epochs (1,420 mini-batches on 100%)
BATCH_SIZE   = 256              # Vectorized parallel mini-batch size


def compute_normalized_laplacian(adj_mx: np.ndarray) -> torch.Tensor:
    """Computes symmetric normalized Laplacian with self-loops: D^{-1/2} A D^{-1/2}."""
    adj = adj_mx.copy()
    np.fill_diagonal(adj, 1.0)
    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[deg == 0] = 0.0
    d_mat = np.diag(deg_inv_sqrt)
    adj_norm = d_mat @ adj @ d_mat
    return torch.tensor(adj_norm, dtype=torch.float32)


def train_and_eval_gcn(
    X_train_t: torch.Tensor, y_train_t: torch.Tensor,
    X_test_t: torch.Tensor, y_test_mph: np.ndarray,
    adj_norm: torch.Tensor, mean: float, std: float,
    is_delta: bool = True, epochs: int = EPOCHS, lr: float = 0.001,
    batch_size: int = BATCH_SIZE, seed: int = 42
):
    """
    Vectorized PyTorch training and evaluation loop.
    Pre-allocated tensors avoid per-batch allocation overhead.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TrafficGCN(in_channels=X_train_t.shape[2], hidden_channels=64, out_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    n_train = len(X_train_t)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            bx = X_train_t[idx]
            by = y_train_t[idx]

            if is_delta:
                target = by - bx[:, :, -1]
            else:
                target = by

            optimizer.zero_grad()
            pred = model(bx, adj_norm=adj_norm)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

    # Evaluation on test set
    model.eval()
    preds_list = []
    with torch.no_grad():
        for i in range(0, len(X_test_t), batch_size):
            bx = X_test_t[i:i + batch_size]
            pred = model(bx, adj_norm=adj_norm)
            if is_delta:
                pred_norm = bx[:, :, -1] + pred
            else:
                pred_norm = pred
            pred_mph = pred_norm.numpy() * std + mean
            preds_list.append(pred_mph)

    pred_all = np.vstack(preds_list)
    mae, rmse = calculate_metrics(y_test_mph, pred_all)
    return mae, rmse, model


def run_regime(
    X_train_g, y_train_g, X_test_g, y_test_g,
    X_train_m, y_train_m, X_test_m, y_test_m,
    adj_norm, mean, std, frac, label
):
    """
    Trains and evaluates all models for one data-availability regime.
    """
    X_sub_g, y_sub_g = get_data_subset(X_train_g, y_train_g, frac)
    X_sub_m, y_sub_m = get_data_subset(X_train_m, y_train_m, frac)
    print(f"\n[{label}] Training samples: {X_sub_g.shape[0]}", flush=True)

    results = {}

    # Pre-convert graph tensors for speed
    X_sub_g_t = torch.tensor(X_sub_g, dtype=torch.float32)
    y_sub_g_t = torch.tensor(y_sub_g, dtype=torch.float32)
    X_test_g_t = torch.tensor(X_test_g, dtype=torch.float32)
    y_test_mph = y_test_g * std + mean

    # ── 1. Linear Regression (Ridge) ──────────────────────────────────────────
    t0 = time.time()
    sub_lr_n = min(10000, len(X_sub_m))
    idx_lr = np.random.choice(len(X_sub_m), size=sub_lr_n, replace=False) if len(X_sub_m) > sub_lr_n else slice(None)
    lr_model = Ridge(alpha=100.0)
    lr_model.fit(X_sub_m[idx_lr], y_sub_m[idx_lr])
    pred_lr = lr_model.predict(X_test_m)
    mae_lr, rmse_lr = calculate_metrics(y_test_m, pred_lr)
    results["Linear Regression"] = (mae_lr, rmse_lr, 0.0)
    print(f"  [LR]         MAE = {mae_lr:.3f} mph  RMSE = {rmse_lr:.3f} mph  ({time.time()-t0:.1f}s)", flush=True)

    # ── 2. Random Forest ──────────────────────────────────────────────────────
    t0 = time.time()
    sub_rf_n = min(5000, len(X_sub_m))
    idx_rf = np.random.choice(len(X_sub_m), size=sub_rf_n, replace=False) if len(X_sub_m) > sub_rf_n else slice(None)
    rf_model = RandomForestRegressor(n_estimators=20, max_depth=10, max_features='sqrt', n_jobs=-1, random_state=42)
    rf_model.fit(X_sub_m[idx_rf], y_sub_m[idx_rf])
    pred_rf = rf_model.predict(X_test_m)
    mae_rf, rmse_rf = calculate_metrics(y_test_m, pred_rf)
    results["Random Forest"] = (mae_rf, rmse_rf, 0.0)
    print(f"  [RF]         MAE = {mae_rf:.3f} mph  RMSE = {rmse_rf:.3f} mph  ({time.time()-t0:.1f}s)", flush=True)

    # ── 3. Direct TrafficGCN (multi-seed) ─────────────────────────────────────
    t0 = time.time()
    mae_gcn_list, rmse_gcn_list = [], []
    for s in range(N_SEEDS):
        m_d, r_d, _ = train_and_eval_gcn(
            X_sub_g_t, y_sub_g_t, X_test_g_t, y_test_mph,
            adj_norm, mean, std, is_delta=False, seed=42 + s
        )
        mae_gcn_list.append(m_d)
        rmse_gcn_list.append(r_d)
    mae_gcn  = float(np.mean(mae_gcn_list))
    rmse_gcn = float(np.mean(rmse_gcn_list))
    std_gcn  = float(np.std(mae_gcn_list))
    results["Direct TrafficGCN"] = (mae_gcn, rmse_gcn, std_gcn)
    print(f"  [GCN-Direct] MAE = {mae_gcn:.3f} ± {std_gcn:.3f} mph  RMSE = {rmse_gcn:.3f} mph  ({time.time()-t0:.1f}s)", flush=True)

    # ── 4. TrafficGCN-Delta (multi-seed) ──────────────────────────────────────
    t0 = time.time()
    mae_delta_list, rmse_delta_list = [], []
    last_delta_model = None
    for s in range(N_SEEDS):
        m_delta, r_delta, last_delta_model = train_and_eval_gcn(
            X_sub_g_t, y_sub_g_t, X_test_g_t, y_test_mph,
            adj_norm, mean, std, is_delta=True, seed=42 + s
        )
        mae_delta_list.append(m_delta)
        rmse_delta_list.append(r_delta)
    mae_delta  = float(np.mean(mae_delta_list))
    rmse_delta = float(np.mean(rmse_delta_list))
    std_delta  = float(np.std(mae_delta_list))
    results["TrafficGCN-Delta (Proposed)"] = (mae_delta, rmse_delta, std_delta)
    print(f"  [GCN-Delta]  MAE = {mae_delta:.3f} ± {std_delta:.3f} mph  RMSE = {rmse_delta:.3f} mph  ({time.time()-t0:.1f}s)", flush=True)

    # Save checkpoint
    ckpt_path = CHECKPOINT_DIR / f"pems_bay_delta_{label.replace('%', 'pct')}.pt"
    torch.save(last_delta_model.state_dict(), ckpt_path)

    return results


def main():
    print("=" * 70, flush=True)
    print("SCRIPT 09: PEMS-BAY CROSS-DATASET BENCHMARK", flush=True)
    print("=" * 70, flush=True)

    # 1. Load PEMS-BAY
    print("\n[1/4] Loading PEMS-BAY dataset ...", flush=True)
    df = load_pems_bay(raw_dir=PEMS_RAW_DIR)
    print(f"      Shape: {df.shape}  |  Sensors: {df.shape[1]}  |  Timesteps: {df.shape[0]}", flush=True)

    # 2. Build Adjacency Matrix
    print("\n[2/4] Building graph adjacency matrix ...", flush=True)
    adj_mx, _ = build_pems_bay_adjacency(raw_dir=PEMS_RAW_DIR)
    adj_norm = compute_normalized_laplacian(adj_mx)
    n_nodes = adj_mx.shape[0]
    n_edges = int(np.sum(adj_mx > 0))
    sparsity = 1.0 - n_edges / (n_nodes ** 2)
    print(f"      Nodes: {n_nodes}  |  Edges: {n_edges}  |  Sparsity: {sparsity:.2%}", flush=True)

    # 3. Preprocess sequences
    print("\n[3/4] Preprocessing sequences ...", flush=True)
    values_norm, mean, std = clean_and_normalize(df, train_ratio=0.70)
    X_graph, y_graph = create_graph_sequences(values_norm, seq_length=SEQ_LENGTH)

    n = len(X_graph)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X_train_g = X_graph[:train_end];  y_train_g = y_graph[:train_end]
    X_test_g  = X_graph[val_end:];    y_test_g  = y_graph[val_end:]

    values_raw = df.values
    X_multi, y_multi = create_multivariate_sequences(values_raw, seq_length=SEQ_LENGTH)
    X_train_m = X_multi[:train_end];  y_train_m = y_multi[:train_end]
    X_test_m  = X_multi[val_end:];    y_test_m  = y_multi[val_end:]

    print(f"      Train samples: {X_train_g.shape[0]}  |  Test samples: {X_test_g.shape[0]}", flush=True)

    # 4. Benchmark across regimes
    print("\n[4/4] Executing benchmark across data regimes ...", flush=True)
    all_results = {}
    for frac, label in REGIMES:
        all_results[label] = run_regime(
            X_train_g, y_train_g, X_test_g, y_test_g,
            X_train_m, y_train_m, X_test_m, y_test_m,
            adj_norm, mean, std, frac, label
        )

    # 5. Compile results table
    rows = []
    model_names = [
        "Linear Regression",
        "Random Forest",
        "Direct TrafficGCN",
        "TrafficGCN-Delta (Proposed)"
    ]
    for mname in model_names:
        row = {"Model": mname}
        for _, label in REGIMES:
            vals = all_results[label].get(mname, (None, None, 0.0))
            row[f"MAE_{label}"] = round(vals[0], 3)
            row[f"MAE_std_{label}"] = round(vals[2], 3)
            row[f"RMSE_{label}"] = round(vals[1], 3)
        mae_100 = row["MAE_100%"]
        mae_20  = row["MAE_20%"]
        if mae_100 and mae_20:
            deg = (mae_20 - mae_100) / mae_100 * 100.0
            row["Degradation_pct"] = round(deg, 2)
        else:
            row["Degradation_pct"] = None
        rows.append(row)

    results_df = pd.DataFrame(rows)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)

    print("\n" + "=" * 70, flush=True)
    print("PEMS-BAY BENCHMARK RESULTS SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(results_df.to_string(index=False), flush=True)
    print(f"\nExported to: {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
