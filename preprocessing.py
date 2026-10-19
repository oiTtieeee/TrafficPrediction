"""
Data preprocessing, sliding window sequence generation, and normalization routines.
"""
import numpy as np
import pandas as pd

def simulate_missing_data(df, missing_rate=0.3, random_state=42):
    """
    Simulates random sensor dropouts / missing observations.
    """
    rng = np.random.default_rng(random_state)
    df_missing = df.copy()
    mask = rng.random(df.shape) < missing_rate
    df_missing[mask] = np.nan
    return df_missing

def impute_missing(df):
    """
    Applies forward-fill followed by backward-fill to restore continuous time-series.
    """
    return df.replace(0, np.nan).ffill().bfill()

def create_sequences(data, window_size=5):
    """
    Univariate autoregressive sequence generator:
    Takes 1D array of length T and yields X: (T - W, W) and y: (T - W,).
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def create_multivariate_sequences(data, seq_length=12):
    """
    Multivariate flattened sequence generator:
    data: (timesteps, num_sensors)
    returns X: (samples, seq_length * num_sensors), y: (samples, num_sensors)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length].flatten())
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def create_graph_sequences(data, seq_length=12):
    """
    3D Spatiotemporal graph sequence generator:
    data: (timesteps, num_sensors)
    returns X: (samples, num_sensors, seq_length), y: (samples, num_sensors)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length].T)
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def clean_and_normalize(df, train_ratio=0.70):
    """
    Applies zero-imputation and strictly normalizes using train-set statistics (mean, std).
    """
    df_clean = impute_missing(df)
    values_clean = df_clean.values
    train_len = int(len(values_clean) * train_ratio)

    mean = float(values_clean[:train_len].mean())
    std = float(values_clean[:train_len].std())
    values_norm = (values_clean - mean) / std

    return values_norm, mean, std

def get_data_subset(X, y, fraction):
    """
    Retrieves the most recent fraction of training data chronologically.
    """
    n = int(len(X) * fraction)
    return X[-n:], y[-n:]
