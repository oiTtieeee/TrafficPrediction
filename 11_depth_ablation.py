import os, sys, time, csv
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import RAW_DIR, CACHE_DIR, SEQ_LENGTH, DEVICE
from src.data_loader import load_metr_la, download_graph_files
from src.graph_utils import build_adjacency
from src.preprocessing import clean_and_normalize, create_graph_sequences
from src.evaluate import calculate_metrics

class NLayerDenseGCN(nn.Module):
    def __init__(self, in_channels=12, hidden_channels=64, out_channels=1, n_layers=2):
        super().__init__()
        self.n_layers = n_layers
        if n_layers == 1:
            self.fc1 = nn.Linear(in_channels, out_channels)
        elif n_layers == 2:
            self.fc1 = nn.Linear(in_channels, hidden_channels)
            self.fc2 = nn.Linear(hidden_channels, out_channels)
        elif n_layers == 3:
            self.fc1 = nn.Linear(in_channels, hidden_channels)
            self.fc2 = nn.Linear(hidden_channels, hidden_channels)
            self.fc3 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, adj_norm):
        # x: (batch_size, num_nodes, in_channels)
        if self.n_layers == 1:
            out = torch.matmul(adj_norm, x)
            out = self.fc1(out)
            return out.squeeze(-1)
        elif self.n_layers == 2:
            h1 = torch.matmul(adj_norm, x)
            h1 = F.relu(self.fc1(h1))
            out = torch.matmul(adj_norm, h1)
            out = self.fc2(out)
            return out.squeeze(-1)
        elif self.n_layers == 3:
            h1 = torch.matmul(adj_norm, x)
            h1 = F.relu(self.fc1(h1))
            h2 = torch.matmul(adj_norm, h1)
            h2 = F.relu(self.fc2(h2))
            out = torch.matmul(adj_norm, h2)
            out = self.fc3(out)
            return out.squeeze(-1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_eval(X_train, y_train, X_val, y_val, X_test, y_test, adj_norm, mean, std, n_layers=2, epochs=30, batch_size=64, lr=0.001):
    torch.manual_seed(42)
    np.random.seed(42)
    model = NLayerDenseGCN(in_channels=X_train.shape[2], hidden_channels=64, out_channels=1, n_layers=n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    n_train = len(X_train)

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            bx = torch.tensor(X_train[idx], dtype=torch.float32)
            by = torch.tensor(y_train[idx], dtype=torch.float32)
            last_val = bx[:, :, -1]
            target_delta = by - last_val
            pred_delta = model(bx, adj_norm)
            loss = criterion(pred_delta, target_delta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation on full test set
    model.eval()
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            bx = torch.tensor(X_test[i:i+batch_size], dtype=torch.float32)
            by = torch.tensor(y_test[i:i+batch_size], dtype=torch.float32)
            last_val = bx[:, :, -1]
            pred_delta = model(bx, adj_norm)
            pred_norm = last_val + pred_delta
            # De-normalize
            pred_mph = pred_norm * std + mean
            true_mph = by * std + mean
            all_preds.append(pred_mph.numpy())
            all_trues.append(true_mph.numpy())

    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    mae, rmse = calculate_metrics(trues, preds)
    params = count_parameters(model)
    return params, mae, rmse

def main():
    print('Loading METR-LA data...')
    df = load_metr_la(raw_dir=RAW_DIR)
    paths = download_graph_files(raw_dir=RAW_DIR)
    with open(paths['graph_sensor_ids.txt']) as f:
        sensor_ids = f.read().strip().split(',')
    dist_df = pd.read_csv(paths['distances_la_2012.csv'], dtype={'from': str, 'to': str})
    adj_mx = build_adjacency(sensor_ids, dist_df)

    adj = adj_mx.copy()
    np.fill_diagonal(adj, 1.0)
    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[deg == 0] = 0.0
    d_mat = np.diag(deg_inv_sqrt)
    adj_norm = torch.tensor(d_mat @ adj @ d_mat, dtype=torch.float32)

    values_norm, mean, std = clean_and_normalize(df, train_ratio=0.70)
    X_graph, y_graph = create_graph_sequences(values_norm, seq_length=SEQ_LENGTH)
    n = len(X_graph)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    X_train, y_train = X_graph[:train_end], y_graph[:train_end]
    X_val, y_val = X_graph[train_end:val_end], y_graph[train_end:val_end]
    X_test, y_test = X_graph[val_end:], y_graph[val_end:]

    results = []
    for n_layers in [1, 2, 3]:
        print(f'Running {n_layers}-layer TrafficGCN-Delta...')
        t0 = time.time()
        params, mae, rmse = train_and_eval(X_train, y_train, X_val, y_val, X_test, y_test, adj_norm, mean, std, n_layers=n_layers)
        print(f'  {n_layers}-layer -> Params: {params:,} | Test MAE: {mae:.4f} mph | RMSE: {rmse:.4f} mph (Time: {time.time()-t0:.1f}s)')
        results.append({'n_layers': n_layers, 'params': params, 'MAE_mph': round(mae, 4), 'RMSE_mph': round(rmse, 4)})

    out_path = PROJECT_ROOT / 'outputs' / 'depth_ablation_results.csv'
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['n_layers', 'params', 'MAE_mph', 'RMSE_mph'])
        writer.writeheader()
        writer.writerows(results)
    print(f'Saved to {out_path}')

if __name__ == '__main__':
    main()
