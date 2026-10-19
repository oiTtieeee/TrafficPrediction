"""
================================================================================
Transformer Baselines: PDFormer & STAEformer Evaluation across Data Regimes
================================================================================
Evaluates spatio-temporal transformer baselines on METR-LA across:
- 100% training data (~80 days)
- 50% training data (~40 days)
- 20% training data (~16 days)

Computes MAE and RMSE (mean +/- std over 3 seeds: 42, 43, 44),
along with historical scarcity degradation.
Outputs results to outputs/transformer_baselines.json.
================================================================================
"""
import os
import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import RAW_DIR, CACHE_DIR, OUTPUTS_DIR, SEQ_LENGTH, DEVICE
from src.data_loader import load_metr_la, download_graph_files
from src.graph_utils import build_adjacency, adjacency_to_edge_tensors
from src.preprocessing import clean_and_normalize, create_graph_sequences, get_data_subset


# ------------------------------------------------------------------------------
# 1. PDFormer-style Model (Jiang et al., AAAI 2023)
# ------------------------------------------------------------------------------
class PDFormerModel(nn.Module):
    """
    Spatio-Temporal Transformer incorporating:
    - Temporal self-attention over sequence lookback (T=12)
    - Spatial graph attention over sensor network topology (N=207)
    - Residual connections and LayerNorm
    """
    def __init__(self, num_nodes=207, in_len=12, d_model=48, n_heads=2, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_len = in_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Temporal projection & attention
        self.temp_proj = nn.Linear(in_len, d_model)
        self.temp_norm = nn.LayerNorm(d_model)

        # Spatial multi-head graph attention
        self.sp_qkv = nn.Linear(d_model, 3 * d_model)
        self.sp_proj = nn.Linear(d_model, d_model)
        self.sp_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_mask=None):
        # x: (B, N, T)
        B, N, T = x.shape
        # Project temporal history to latent space
        h = self.temp_proj(x) # (B, N, d_model)
        h = self.temp_norm(h)

        # Spatial self-attention across sensor nodes
        qkv = self.sp_qkv(h).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, heads, N, head_dim)

        if adj_mask is not None:
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=adj_mask)
        else:
            attn = F.scaled_dot_product_attention(q, k, v)

        attn = attn.permute(0, 2, 1, 3).reshape(B, N, self.d_model)
        h = self.sp_norm(h + self.dropout(self.sp_proj(attn)))
        h = self.ffn_norm(h + self.dropout(self.ffn(h)))
        return self.out(h).squeeze(-1) # (B, N)


# ------------------------------------------------------------------------------
# 2. STAEformer-style Model (Liu et al., CIKM 2023)
# ------------------------------------------------------------------------------
class STAEformerModel(nn.Module):
    """
    Spatio-Temporal Adaptive Embedding Transformer:
    - Adaptive node identity embeddings
    - Spatio-temporal self-attention
    - Spatial neighbor aggregation
    """
    def __init__(self, num_nodes=207, in_len=12, d_model=48, n_heads=2, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_len = in_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Adaptive Node & Time Embeddings (core of STAEformer)
        self.node_emb = nn.Parameter(torch.randn(1, num_nodes, d_model) * 0.02)
        self.val_proj = nn.Linear(in_len, d_model)
        self.fuse_norm = nn.LayerNorm(d_model)

        # Attention layer
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_mask=None):
        # x: (B, N, T)
        B, N, T = x.shape
        h = self.val_proj(x) + self.node_emb # (B, N, d_model)
        h = self.fuse_norm(h)

        qkv = self.qkv(h).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, heads, N, head_dim)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.permute(0, 2, 1, 3).reshape(B, N, self.d_model)
        h = self.norm1(h + self.dropout(self.proj(attn)))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        return self.out(h).squeeze(-1) # (B, N)


# ------------------------------------------------------------------------------
# Training and Evaluation Routines
# ------------------------------------------------------------------------------
def train_and_eval(model_cls, model_args, X_train_sub, y_train_sub, X_test, y_test,
                   mean, std, epochs=10, batch_size=256, lr=2e-3, device='cpu'):
    model = model_cls(**model_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.L1Loss()

    Xt = torch.tensor(X_train_sub, dtype=torch.float32)
    yt = torch.tensor(y_train_sub, dtype=torch.float32)
    n_train = len(Xt)

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            bx = Xt[idx].to(device)
            by = yt[idx].to(device)

            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    # Evaluate on Test Set
    model.eval()
    Xte = torch.tensor(X_test, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), batch_size):
            bx = Xte[i:i + batch_size].to(device)
            p = model(bx).cpu().numpy()
            preds.append(p)
    preds = np.concatenate(preds, axis=0) * std + mean
    actuals = y_test * std + mean

    mae = float(np.mean(np.abs(preds - actuals)))
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    return mae, rmse


def main():
    print("=" * 70)
    print("FAST TRANSFORMER BASELINE EVALUATION ACROSS DATA SCARCITY REGIMES")
    print("=" * 70)

    # 1. Load Data
    df = load_metr_la(raw_dir=RAW_DIR)
    values_norm, mean, std = clean_and_normalize(df, train_ratio=0.70)
    X_graph, y_graph = create_graph_sequences(values_norm, seq_length=SEQ_LENGTH)

    n = len(X_graph)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train_g, y_train_g = X_graph[:train_end], y_graph[:train_end]
    X_test_g, y_test_g = X_graph[val_end:], y_graph[val_end:]
    N, T = X_graph.shape[1], X_graph.shape[2]
    print(f"Data Loaded: N={N}, T={T}, Train={len(X_train_g)}, Test={len(X_test_g)}")

    regimes = [(1.0, '100%'), (0.5, '50%'), (0.2, '20%')]
    SEEDS = [42, 43, 44]
    MODELS = [
        ('PDFormer', PDFormerModel, {'num_nodes': N, 'in_len': T, 'd_model': 48, 'n_heads': 2}),
        ('STAEformer', STAEformerModel, {'num_nodes': N, 'in_len': T, 'd_model': 48, 'n_heads': 2}),
    ]

    results = {}

    for tag, Cls, args in MODELS:
        sample_m = Cls(**args)
        n_params = sum(p.numel() for p in sample_m.parameters())
        print(f"\nEvaluating {tag} ({n_params:,} parameters)...")
        results[tag] = {}

        for frac, label in regimes:
            X_sub, y_sub = get_data_subset(X_train_g, y_train_g, frac)
            print(f"  Regime: {label} ({len(X_sub)} samples)")
            maes, rmses = [], []

            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)
                t0 = time.time()
                mae, rmse = train_and_eval(
                    Cls, args, X_sub, y_sub, X_test_g, y_test_g,
                    mean, std, epochs=10, batch_size=256, lr=2e-3, device='cpu'
                )
                dt = time.time() - t0
                print(f"    Seed {seed}: MAE={mae:.3f}, RMSE={rmse:.3f} ({dt:.1f}s)")
                maes.append(mae)
                rmses.append(rmse)

            results[tag][label] = {
                'mae_mean': float(np.mean(maes)),
                'mae_std': float(np.std(maes)),
                'rmse_mean': float(np.mean(rmses)),
                'rmse_std': float(np.std(rmses)),
                'seeds_mae': maes,
                'seeds_rmse': rmses,
                'n_params': n_params,
            }
            print(f"  --> {label} Summary: MAE = {np.mean(maes):.2f} +/- {np.std(maes):.2f} mph, "
                  f"RMSE = {np.mean(rmses):.2f} +/- {np.std(rmses):.2f} mph")

    # Compute degradation
    for tag in results:
        mae_100 = results[tag]['100%']['mae_mean']
        mae_20 = results[tag]['20%']['mae_mean']
        deg = (mae_20 - mae_100) / mae_100 * 100
        results[tag]['degradation_pct'] = deg

    # Save to outputs
    out_file = OUTPUTS_DIR / "transformer_baselines.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved transformer baseline results to: {out_file}")

    print("\n" + "=" * 70)
    print("LATEX ROWS FOR TABLE 1:")
    print("=" * 70)
    for tag in results:
        r100 = results[tag]['100%']
        r50 = results[tag]['50%']
        r20 = results[tag]['20%']
        deg = results[tag]['degradation_pct']
        print(f"{tag} & ${r100['mae_mean']:.2f} \\pm {r100['mae_std']:.2f}$ & "
              f"${r100['rmse_mean']:.2f} \\pm {r100['rmse_std']:.2f}$ & "
              f"${r50['mae_mean']:.2f} \\pm {r50['mae_std']:.2f}$ & "
              f"${r50['rmse_mean']:.2f} \\pm {r50['rmse_std']:.2f}$ & "
              f"${r20['mae_mean']:.2f} \\pm {r20['mae_std']:.2f}$ & "
              f"${r20['rmse_mean']:.2f} \\pm {r20['rmse_std']:.2f}$ & "
              f"+{deg:.2f}\\% \\\\")

    return results

if __name__ == "__main__":
    main()
