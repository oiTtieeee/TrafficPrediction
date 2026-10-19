"""
Graph Convolutional Network (TrafficGCN) model architecture using PyTorch Geometric.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv
except ImportError:
    # Graceful fallback or warning
    GCNConv = None

class TrafficGCN(nn.Module):
    """
    Two-layer Graph Convolutional Network for spatial node feature aggregation:
        conv1: (in_channels -> hidden_channels) + ReLU
        conv2: (hidden_channels -> out_channels)
    Supports both PyTorch Geometric (edge_index, edge_weight) and
    pure PyTorch dense normalized adjacency matrix (N, N).
    """
    def __init__(self, in_channels, hidden_channels=64, out_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        if GCNConv is not None:
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
            self.use_pyg = True
        else:
            self.use_pyg = False
            self.fc1 = nn.Linear(in_channels, hidden_channels)
            self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index=None, edge_weight=None, adj_norm=None):
        """
        x: (num_nodes, in_channels) or (batch_size, num_nodes, in_channels)
        edge_index: (2, num_edges) optional for PyG
        edge_weight: (num_edges,) optional for PyG
        adj_norm: (num_nodes, num_nodes) normalized adjacency matrix
        """
        if self.use_pyg and edge_index is not None and adj_norm is None:
            x = self.conv1(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.conv2(x, edge_index, edge_weight)
            return x.squeeze(-1)
        else:
            # Pure PyTorch vectorized dense GCN implementation
            # Layer 1: H1 = ReLU( adj_norm @ x @ W0 + b0 )
            h1 = torch.matmul(adj_norm, x)
            h1 = F.relu(self.fc1(h1))
            # Layer 2: out = adj_norm @ H1 @ W1 + b1
            out = torch.matmul(adj_norm, h1)
            out = self.fc2(out)
            return out.squeeze(-1)
