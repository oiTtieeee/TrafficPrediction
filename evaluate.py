"""
Evaluation metrics, de-normalization, and results reporting utilities.
"""
import csv
import numpy as np
import torch
from pathlib import Path
from src.train import make_batch, make_batch_delta
from src.config import RESULTS_CSV

def calculate_metrics(actual, pred):
    """
    Computes Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
    """
    mae = float(np.mean(np.abs(pred - actual)))
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    return mae, rmse

def evaluate_gcn(model, X_np, y_np, edge_index, edge_weight, mean, std, device, batch_size=64):
    """
    Evaluates standard GCN model on test split, reconstructing speeds to original scale.
    """
    model.eval()
    preds_all, actuals_all = [], []
    with torch.no_grad():
        for i in range(0, X_np.shape[0], batch_size):
            idx = np.arange(i, min(i + batch_size, X_np.shape[0]))
            batch = make_batch(X_np, y_np, idx, edge_index, edge_weight).to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            preds_all.append(out.cpu().numpy())
            actuals_all.append(batch.y.cpu().numpy())

    preds = np.concatenate(preds_all) * std + mean
    actuals = np.concatenate(actuals_all) * std + mean
    mae, rmse = calculate_metrics(actuals, preds)
    return mae, rmse, preds

def evaluate_gcn_delta(model, X_np, y_np, edge_index, edge_weight, mean, std, device, batch_size=64):
    """
    Evaluates TrafficGCN-Delta model, reconstructing true speed from delta and de-normalizing.
    """
    model.eval()
    preds_all, actuals_all = [], []
    with torch.no_grad():
        for i in range(0, X_np.shape[0], batch_size):
            idx = np.arange(i, min(i + batch_size, X_np.shape[0]))
            batch = make_batch_delta(X_np, y_np, idx, edge_index, edge_weight).to(device)
            delta_pred = model(batch.x, batch.edge_index, batch.edge_attr)
            pred = batch.last_val + delta_pred
            actual = batch.last_val + batch.y
            preds_all.append(pred.cpu().numpy())
            actuals_all.append(actual.cpu().numpy())

    preds = np.concatenate(preds_all) * std + mean
    actuals = np.concatenate(actuals_all) * std + mean
    mae, rmse = calculate_metrics(actuals, preds)
    return mae, rmse, preds

def append_result_row(model_name, data_pct, mae, rmse, results_csv=None):
    """
    Logs an experimental run to results.csv.
    """
    results_csv = Path(results_csv or RESULTS_CSV)
    file_exists = results_csv.exists()
    with open(results_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model", "data_pct", "mae", "rmse"])
        writer.writerow([model_name, data_pct, round(float(mae), 3), round(float(rmse), 3)])
    print(f"Logged result for {model_name} @ {data_pct} to {results_csv}")
