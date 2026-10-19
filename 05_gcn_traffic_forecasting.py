"""
================================================================================
Section 9-11: Graph Sequence Preparation & Spatiotemporal GCN Modeling
================================================================================
This script:
1. Imputes missing values and strictly standardizes speeds using training stats.
2. Extracts 3D graph sequences: (N_samples, 207 sensors, 12 lookback timesteps).
3. Constructs PyTorch Geometric edge tensors from the road network graph.
4. Implements the TrafficGCN 2-layer Graph Convolutional Network.
5. Trains standard TrafficGCN on direct normalized speeds.
6. Implements the TrafficGCN-Delta residual velocity change formulation.
7. Evaluates both architectures on the test split, de-normalizing predictions to original mph.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from src.config import RAW_DIR, CACHE_DIR, SEQ_LENGTH, DEVICE
from src.data_loader import load_metr_la, download_graph_files
from src.graph_utils import build_adjacency, adjacency_to_edge_tensors
from src.preprocessing import clean_and_normalize, create_graph_sequences
from src.models.gcn import TrafficGCN
from src.train import train_gcn, train_gcn_delta
from src.evaluate import evaluate_gcn, evaluate_gcn_delta

def main():
    print("=" * 70)
    print("SECTION 9: GRAPH SEQUENCE PREPARATION & LEAK-FREE STANDARDIZATION")
    print("=" * 70)

    # 1. Load Data
    df = load_metr_la(raw_dir=RAW_DIR)
    print(f"Loaded traffic speed data: {df.shape}")

    # 2. Graph topology
    graph_tensors_path = CACHE_DIR / "graph_tensors.pt"
    if graph_tensors_path.exists():
        print(f"Loading cached graph tensors from {graph_tensors_path}...")
        graph_cache = torch.load(graph_tensors_path, weights_only=False)
        edge_index = graph_cache["edge_index"]
        edge_weight = graph_cache["edge_weight"]
    else:
        print("Computing graph tensors from raw metadata...")
        paths = download_graph_files(raw_dir=RAW_DIR)
        with open(paths["graph_sensor_ids.txt"]) as f:
            sensor_ids = f.read().strip().split(',')
        dist_df = pd.read_csv(paths["distances_la_2012.csv"], dtype={'from': str, 'to': str})
        adj_mx = build_adjacency(sensor_ids, dist_df)
        edge_index, edge_weight = adjacency_to_edge_tensors(adj_mx)

    print(f"Graph connectivity -> edge_index: {edge_index.shape}, edge_weight: {edge_weight.shape}")

    # 3. Clean & Normalize using train-set statistics strictly
    print("\nCleaning data and computing train-set normalization parameters...")
    values_norm, mean, std = clean_and_normalize(df, train_ratio=0.70)
    print(f"Normalization Stats: Mean = {mean:.4f}, Std = {std:.4f}")

    # 4. Spatiotemporal sequences
    print(f"\nBuilding 3D graph sequences (SEQ_LENGTH = {SEQ_LENGTH})...")
    X_graph, y_graph = create_graph_sequences(values_norm, seq_length=SEQ_LENGTH)
    print(f"Graph sequences shape: X = {X_graph.shape}, y = {y_graph.shape}")

    # 5. Chronological Partition: 70% Train, 15% Val, 15% Test
    n = len(X_graph)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train_g, y_train_g = X_graph[:train_end], y_graph[:train_end]
    X_val_g, y_val_g     = X_graph[train_end:val_end], y_graph[train_end:val_end]
    X_test_g, y_test_g   = X_graph[val_end:], y_graph[val_end:]

    print(f"Split sizes -> Train: {X_train_g.shape[0]} | Val: {X_val_g.shape[0]} | Test: {X_test_g.shape[0]}")

    # 6. Section 10: Direct GCN Model
    print("\n" + "=" * 70)
    print("SECTION 10: DIRECT GRAPH CONVOLUTIONAL NETWORK (TRAFFICGCN)")
    print("=" * 70)
    print(f"Using compute device: {DEVICE}")

    print("\nTraining direct TrafficGCN (20 Epochs, lr=0.01)...")
    direct_model = train_gcn(
        X_train_g, y_train_g, X_val_g, y_val_g,
        edge_index, edge_weight,
        epochs=20, batch_size=64, lr=0.01, device=DEVICE
    )

    print("\nEvaluating direct TrafficGCN on Test Set...")
    mae_direct, rmse_direct, _ = evaluate_gcn(
        direct_model, X_test_g, y_test_g,
        edge_index, edge_weight, mean, std, device=DEVICE
    )
    print(f"    Direct TrafficGCN -> MAE: {mae_direct:.3f} mph, RMSE: {rmse_direct:.3f} mph")

    # 7. Section 11: Residual / Momentum Delta Formulation
    print("\n" + "=" * 70)
    print("SECTION 11: RESIDUAL CHANGE FORMULATION (TRAFFICGCN-DELTA)")
    print("=" * 70)

    print("\nTraining TrafficGCN-Delta (predicting velocity change Delta y, 30 Epochs, lr=0.001)...")
    delta_model = train_gcn_delta(
        X_train_g, y_train_g, X_val_g, y_val_g,
        edge_index, edge_weight,
        epochs=30, batch_size=64, lr=0.001, device=DEVICE
    )

    print("\nEvaluating TrafficGCN-Delta on Test Set (Reconstructing true speed)...")
    mae_delta, rmse_delta, _ = evaluate_gcn_delta(
        delta_model, X_test_g, y_test_g,
        edge_index, edge_weight, mean, std, device=DEVICE
    )
    print(f"    TrafficGCN-Delta  -> MAE: {mae_delta:.3f} mph, RMSE: {rmse_delta:.3f} mph")

    print("\nDone! Section 9-11 completed successfully.")

if __name__ == "__main__":
    main()
