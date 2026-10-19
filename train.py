"""
Batch generation and training routines for PyTorch Geometric Graph Neural Networks.
"""
import numpy as np
import torch
import torch.nn as nn
from src.models.gcn import TrafficGCN

try:
    from torch_geometric.data import Data, Batch
except ImportError:
    Data, Batch = None, None

def make_batch(X_np, y_np, indices, edge_index, edge_weight):
    """
    Constructs a PyTorch Geometric Batch from temporal graph slices.
    """
    if Data is None or Batch is None:
        raise ImportError("torch_geometric is required for batch creation.")
    data_list = []
    for i in indices:
        x = torch.tensor(X_np[i], dtype=torch.float)
        y = torch.tensor(y_np[i], dtype=torch.float)
        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y))
    return Batch.from_data_list(data_list)

def make_batch_delta(X_np, y_np, indices, edge_index, edge_weight):
    """
    Constructs a PyTorch Geometric Batch predicting residual changes: y_delta = y - last_val.
    """
    if Data is None or Batch is None:
        raise ImportError("torch_geometric is required for batch creation.")
    data_list = []
    for i in indices:
        x = torch.tensor(X_np[i], dtype=torch.float)
        last_val = x[:, -1]
        y_delta = torch.tensor(y_np[i], dtype=torch.float) - last_val
        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_weight,
                              y=y_delta, last_val=last_val))
    return Batch.from_data_list(data_list)

def train_gcn(X_train_g, y_train_g, X_val_g, y_val_g, edge_index, edge_weight,
              epochs=20, batch_size=64, lr=0.01, hidden=64, device=None, verbose=True):
    """
    Trains standard TrafficGCN predicting absolute normalized speeds.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TrafficGCN(in_channels=X_train_g.shape[2], hidden_channels=hidden, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    edge_index_d = edge_index.to(device)
    edge_weight_d = edge_weight.to(device)
    n_train = X_train_g.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)
        total_loss = 0.0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            batch = make_batch(X_train_g, y_train_g, idx, edge_index_d, edge_weight_d).to(device)

            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        avg_train_loss = total_loss / n_train

        # Validation sample check
        model.eval()
        with torch.no_grad():
            val_idx = np.random.choice(X_val_g.shape[0], size=min(500, X_val_g.shape[0]), replace=False)
            val_batch = make_batch(X_val_g, y_val_g, val_idx, edge_index_d, edge_weight_d).to(device)
            val_out = model(val_batch.x, val_batch.edge_index, val_batch.edge_attr)
            val_loss = criterion(val_out, val_batch.y).item()

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} - Train MAE: {avg_train_loss:.4f} | Val MAE (sample): {val_loss:.4f}")

    return model

def train_gcn_delta(X_train_g, y_train_g, X_val_g, y_val_g, edge_index, edge_weight,
                    epochs=30, batch_size=64, lr=0.001, hidden=64, device=None, verbose=True):
    """
    Trains TrafficGCN on residual momentum: Delta y = y - x[:,-1].
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TrafficGCN(in_channels=X_train_g.shape[2], hidden_channels=hidden, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    edge_index_d = edge_index.to(device)
    edge_weight_d = edge_weight.to(device)
    n_train = X_train_g.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)
        total_loss = 0.0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            batch = make_batch_delta(X_train_g, y_train_g, idx, edge_index_d, edge_weight_d).to(device)

            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        avg_train_loss = total_loss / n_train

        model.eval()
        with torch.no_grad():
            val_idx = np.random.choice(X_val_g.shape[0], size=min(500, X_val_g.shape[0]), replace=False)
            val_batch = make_batch_delta(X_val_g, y_val_g, val_idx, edge_index_d, edge_weight_d).to(device)
            val_out = model(val_batch.x, val_batch.edge_index, val_batch.edge_attr)
            val_loss = criterion(val_out, val_batch.y).item()

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} - Train MAE(delta): {avg_train_loss:.4f} | Val MAE(delta): {val_loss:.4f}")

    return model
