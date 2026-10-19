import sys
sys.path.insert(0, '.')
import time
import numpy as np
import torch
import torch.nn as nn
from src.config import RAW_DIR
from src.data_loader_pems import load_pems_bay, build_pems_bay_adjacency
from src.preprocessing import clean_and_normalize, create_graph_sequences, get_data_subset
from src.models.gcn import TrafficGCN
from src.evaluate import calculate_metrics

df = load_pems_bay(raw_dir=RAW_DIR)
adj_mx, _ = build_pems_bay_adjacency(raw_dir=RAW_DIR)

adj = adj_mx.copy()
np.fill_diagonal(adj, 1.0)
deg = np.sum(adj, axis=1)
deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
deg_inv_sqrt[deg == 0] = 0.0
d_mat = np.diag(deg_inv_sqrt)
adj_norm = torch.tensor(d_mat @ adj @ d_mat, dtype=torch.float32)

values_norm, mean, std = clean_and_normalize(df, train_ratio=0.70)
X_graph, y_graph = create_graph_sequences(values_norm, seq_length=12)

train_end = int(len(X_graph) * 0.70)
val_end   = int(len(X_graph) * 0.85)

X_train = X_graph[:train_end]
y_train = y_graph[:train_end]
X_test  = X_graph[val_end:]
y_test  = y_graph[val_end:]

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# Fast 1 epoch test
model = TrafficGCN(in_channels=12, hidden_channels=64, out_channels=1)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
crit = nn.L1Loss()

t0 = time.time()
perm = np.random.permutation(len(X_train))
for i in range(0, min(len(X_train), 1000), 64):
    idx = perm[i:i+64]
    bx = torch.tensor(X_train[idx], dtype=torch.float32)
    by = torch.tensor(y_train[idx], dtype=torch.float32)
    last = bx[:, :, -1]
    target = by - last
    pred = model(bx, adj_norm=adj_norm)
    loss = crit(pred, target)
    opt.zero_grad()
    loss.backward()
    opt.step()

print(f"1000 samples step took {time.time()-t0:.2f}s")

# Test evaluation on 1000 test samples
bx_test = torch.tensor(X_test[:1000], dtype=torch.float32)
by_test = torch.tensor(y_test[:1000], dtype=torch.float32)
model.eval()
with torch.no_grad():
    pred_delta = model(bx_test, adj_norm=adj_norm)
    pred_norm = bx_test[:, :, -1] + pred_delta
    pred_mph = pred_norm.numpy() * std + mean
    y_mph = by_test.numpy() * std + mean
    mae, rmse = calculate_metrics(y_mph, pred_mph)
print(f"Eval MAE: {mae:.3f} mph, RMSE: {rmse:.3f} mph")
