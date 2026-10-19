"""
================================================================================
Section 6 & 7: Road Network Topology & Graph Adjacency Construction
================================================================================
This script:
1. Downloads and loads sensor distance and location metadata.
2. Computes the pairwise distance matrix across 207 sensors.
3. Constructs the thresholded Gaussian distance kernel adjacency matrix A.
4. Generates and saves a geospatial plot of the sensor network graph.
5. Saves adjacency matrix and PyTorch Geometric edge tensors to cache.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from src.config import RAW_DIR, CACHE_DIR, OUTPUTS_DIR
from src.data_loader import download_graph_files
from src.graph_utils import build_adjacency, adjacency_to_edge_tensors

plt.style.use('ggplot')

def main():
    print("=" * 70)
    print("SECTION 6: SENSOR TOPOLOGY & ADJACENCY MATRIX CONSTRUCTION")
    print("=" * 70)

    # 1. Download / Load graph metadata
    print("\n[1] Checking and downloading graph files...")
    paths = download_graph_files(raw_dir=RAW_DIR)
    for fname, p in paths.items():
        print(f"    {fname} -> {p}")

    # 2. Parse sensor IDs
    with open(paths["graph_sensor_ids.txt"]) as f:
        sensor_ids = f.read().strip().split(',')
    num_sensors = len(sensor_ids)
    print(f"\n[2] Loaded {num_sensors} sensor IDs.")

    # 3. Build Adjacency Matrix
    print("\n[3] Computing Gaussian distance kernel adjacency matrix...")
    dist_df = pd.read_csv(paths["distances_la_2012.csv"], dtype={'from': str, 'to': str})
    adj_mx = build_adjacency(sensor_ids, dist_df, threshold=0.1)

    print(f"    Adjacency Matrix Shape: {adj_mx.shape}")
    num_edges = int((adj_mx > 0).sum())
    print(f"    Total directed edges (including self-loops): {num_edges}")
    sparsity = 1.0 - (num_edges / (num_sensors * num_sensors))
    print(f"    Graph Sparsity: {sparsity * 100:.2f}%")

    # 4. Convert to PyG Edge Tensors
    edge_index, edge_weight = adjacency_to_edge_tensors(adj_mx)
    print(f"    PyG edge_index tensor shape: {edge_index.shape}")
    print(f"    PyG edge_weight tensor shape: {edge_weight.shape}")

    # Save to Cache
    np.save(CACHE_DIR / "adj_mx.npy", adj_mx)
    torch.save({"edge_index": edge_index, "edge_weight": edge_weight, "sensor_ids": sensor_ids},
               CACHE_DIR / "graph_tensors.pt")
    print(f"    Saved adjacency matrix and edge tensors to {CACHE_DIR}")

    # 5. Geospatial Sensor Layout Plot
    print("\n" + "=" * 70)
    print("SECTION 7: GEOSPATIAL SENSOR NETWORK VISUALIZATION")
    print("=" * 70)

    loc_df = pd.read_csv(paths["graph_sensor_locations.csv"])
    loc_df['sensor_id'] = loc_df['sensor_id'].astype(str)
    loc_df = loc_df.set_index('sensor_id').loc[sensor_ids].reset_index()

    plt.figure(figsize=(10, 10))
    for i in range(num_sensors):
        for j in range(num_sensors):
            if adj_mx[i, j] > 0 and i != j:
                plt.plot([loc_df.longitude[i], loc_df.longitude[j]],
                         [loc_df.latitude[i], loc_df.latitude[j]],
                         'steelblue', linewidth=0.4, alpha=0.5)
    plt.scatter(loc_df.longitude, loc_df.latitude, s=15, c='crimson', zorder=5, label="Traffic Sensors")
    plt.title("METR-LA Sensor Highway Network Topology (Los Angeles)", fontsize=13)
    plt.xlabel("Longitude", fontsize=11)
    plt.ylabel("Latitude", fontsize=11)
    plt.legend(loc="upper right")
    plot_map = OUTPUTS_DIR / "sensor_network_topology.png"
    plt.tight_layout()
    plt.savefig(plot_map, dpi=200)
    plt.close()
    print(f"Saved geospatial graph visualization to: {plot_map}")

    print("\nDone! Section 6 & 7 completed successfully.")

if __name__ == "__main__":
    main()
