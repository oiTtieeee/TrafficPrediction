"""
Graph construction and PyTorch Geometric tensor formatting utilities.
"""
import numpy as np
import pandas as pd
import torch

def build_adjacency(sensor_ids, dist_df, threshold=0.1):
    """
    Constructs a Gaussian distance kernel adjacency matrix with self-loops:
        W_ij = exp(-(dist_ij / std)^2), with values < threshold set to 0.
    """
    num_sensors = len(sensor_ids)
    id_to_idx = {str(sid): i for i, sid in enumerate(sensor_ids)}
    dist_mx = np.full((num_sensors, num_sensors), np.inf)

    froms = dist_df['from'].astype(str).values
    tos = dist_df['to'].astype(str).values
    costs = dist_df['cost'].values

    for f, t, c in zip(froms, tos, costs):
        if f in id_to_idx and t in id_to_idx:
            dist_mx[id_to_idx[f], id_to_idx[t]] = c

    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    adj_mx[adj_mx < threshold] = 0.0
    np.fill_diagonal(adj_mx, 1.0)
    return adj_mx

def adjacency_to_edge_tensors(adj_mx):
    """
    Converts a dense adjacency matrix into PyTorch Geometric edge_index (2, E)
    and edge_weight (E,) tensors.
    """
    edge_index_np = np.array(np.nonzero(adj_mx))
    edge_weight_np = adj_mx[edge_index_np[0], edge_index_np[1]]

    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight_np, dtype=torch.float)
    return edge_index, edge_weight
