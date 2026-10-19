"""
================================================================================
Section 4 & 5: Single-Sensor Autoregressive Baselines (Linear Regression & LSTM)
================================================================================
This script:
1. Generates univariate sliding-window lag sequences (window_size=5 for LR, 12 for LSTM).
2. Fits an autoregressive Ordinary Least Squares (OLS) Linear Regression model.
3. Evaluates Linear Regression on clean and imputed missing data (MAE & RMSE).
4. Fits a Long Short-Term Memory (LSTM) recurrent neural network.
5. Evaluates and plots actual vs predicted traffic speeds.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from src.config import RAW_DIR, OUTPUTS_DIR
from src.data_loader import load_metr_la
from src.preprocessing import create_sequences, simulate_missing_data
from src.models.baselines import train_linear_regression, build_lstm_model
from src.evaluate import calculate_metrics

plt.style.use('ggplot')

def main():
    print("=" * 70)
    print("SECTION 4: SINGLE-SENSOR CLASSICAL BASELINE (LINEAR REGRESSION)")
    print("=" * 70)

    df = load_metr_la(raw_dir=RAW_DIR)
    sensor_id = df.columns[0]
    series = df[sensor_id].values
    print(f"Target Sensor ID: {sensor_id} (Length: {len(series)})")

    # 1. Autoregressive sequence generation (W = 5)
    WINDOW_SIZE = 5
    X, y = create_sequences(series, window_size=WINDOW_SIZE)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")

    # 2. Linear Regression fitting
    print("\nTraining Linear Regression baseline...")
    lr_model = train_linear_regression(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    mae_lr, rmse_lr = calculate_metrics(y_test, y_pred)
    print(f"Linear Regression -> MAE: {mae_lr:.3f}, RMSE: {rmse_lr:.3f}")

    # Plot Linear Regression predictions
    plt.figure(figsize=(12, 5))
    plt.plot(y_test[:200], label='Actual Speed', color='black', lw=1.2)
    plt.plot(y_pred[:200], label='Predicted (Linear Regression)', color='royalblue', linestyle='--', lw=1.2)
    plt.legend()
    plt.title(f"Traffic Speed Prediction: Linear Regression (Sensor {sensor_id})", fontsize=13)
    plt.xlabel("Timestep", fontsize=11)
    plt.ylabel("Speed (mph)", fontsize=11)
    plot_lr = OUTPUTS_DIR / "single_sensor_lr_prediction.png"
    plt.tight_layout()
    plt.savefig(plot_lr, dpi=150)
    plt.close()
    print(f"Saved Linear Regression prediction plot to: {plot_lr}")

    # 3. Test on Missing Data
    print("\nEvaluating Linear Regression on imputed 30% missing data...")
    df_missing = simulate_missing_data(df, missing_rate=0.3, random_state=42)
    series_missing = df_missing[sensor_id].ffill().bfill().values
    X_m, y_m = create_sequences(series_missing, window_size=WINDOW_SIZE)
    split_m = int(0.8 * len(X_m))
    X_train_m, X_test_m = X_m[:split_m], X_m[split_m:]
    y_train_m, y_test_m = y_m[:split_m], y_m[split_m:]

    lr_m = train_linear_regression(X_train_m, y_train_m)
    y_pred_m = lr_m.predict(X_test_m)
    mae_m, rmse_m = calculate_metrics(y_test_m, y_pred_m)
    print(f"Linear Regression (with Missing Data) -> MAE: {mae_m:.3f}, RMSE: {rmse_m:.3f}")

    # 4. Deep Learning Baseline: LSTM
    print("\n" + "=" * 70)
    print("SECTION 5: SINGLE-SENSOR DEEP LEARNING BASELINE (LSTM)")
    print("=" * 70)

    try:
        scaler = MinMaxScaler()
        series_scaled = scaler.fit_transform(series.reshape(-1, 1))

        SEQ_LENGTH = 12  # 1 hour lookback
        X_lstm, y_lstm = create_sequences(series_scaled, window_size=SEQ_LENGTH)
        split_l = int(0.8 * len(X_lstm))
        X_train_l, X_test_l = X_lstm[:split_l], X_lstm[split_l:]
        y_train_l, y_test_l = y_lstm[:split_l], y_lstm[split_l:]

        print(f"X_train_lstm: {X_train_l.shape} | X_test_lstm: {X_test_l.shape}")
        print("Constructing Keras Sequential LSTM model...")
        lstm_model = build_lstm_model(seq_length=SEQ_LENGTH, hidden_units=64, output_dim=1)
        
        print("Training LSTM for 10 epochs...")
        history = lstm_model.fit(
            X_train_l, y_train_l,
            epochs=10,
            batch_size=32,
            validation_split=0.1,
            verbose=1
        )

        # Plot loss curve
        plt.figure(figsize=(8, 5))
        plt.plot(history.history["loss"], label="Training Loss", color='steelblue', lw=1.5)
        plt.plot(history.history["val_loss"], label="Validation Loss", color='crimson', lw=1.5)
        plt.legend()
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Loss (MSE)", fontsize=11)
        plt.title("LSTM Training & Validation Loss Curve", fontsize=13)
        plot_loss = OUTPUTS_DIR / "single_sensor_lstm_loss.png"
        plt.tight_layout()
        plt.savefig(plot_loss, dpi=150)
        plt.close()
        print(f"Saved LSTM loss curve to: {plot_loss}")

        # Out-of-sample evaluation
        pred_scaled = lstm_model.predict(X_test_l)
        pred_lstm = scaler.inverse_transform(pred_scaled)
        actual_lstm = scaler.inverse_transform(y_test_l)

        mae_lstm, rmse_lstm = calculate_metrics(actual_lstm, pred_lstm)
        print(f"LSTM -> MAE: {mae_lstm:.3f}, RMSE: {rmse_lstm:.3f}")

        # Plot actual vs predicted
        plt.figure(figsize=(12, 5))
        plt.plot(actual_lstm[:200], label="Actual Speed", color='black', lw=1.2)
        plt.plot(pred_lstm[:200], label="Predicted (LSTM)", color='forestgreen', linestyle='--', lw=1.2)
        plt.legend()
        plt.title(f"Traffic Speed Prediction: LSTM (Sensor {sensor_id})", fontsize=13)
        plt.xlabel("Timestep", fontsize=11)
        plt.ylabel("Speed (mph)", fontsize=11)
        plot_pred = OUTPUTS_DIR / "single_sensor_lstm_prediction.png"
        plt.tight_layout()
        plt.savefig(plot_pred, dpi=150)
        plt.close()
        print(f"Saved LSTM prediction plot to: {plot_pred}")

    except ImportError as e:
        print(f"Skipping LSTM training (TensorFlow not installed in current environment: {e})")

    print("\nDone! Section 4 & 5 completed successfully.")

if __name__ == "__main__":
    main()
