"""
================================================================================
Section 1-3: METR-LA Data Ingestion, EDA & Missing Value Imputation
================================================================================
This script:
1. Configures environment and imports scientific libraries.
2. Ingests the METR-LA traffic speed dataset (207 sensors x 34,272 timesteps).
3. Computes summary statistics, missing value rates, and saves time-series plots.
4. Simulates a 30% sensor dropout rate and evaluates forward-fill imputation.
"""
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import RAW_DIR, OUTPUTS_DIR
from src.data_loader import load_metr_la
from src.preprocessing import simulate_missing_data

plt.style.use('ggplot')

def main():
    print("=" * 70)
    print("SECTION 1 & 2: METR-LA DATA INGESTION & EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # 1. Ingest Dataset
    print(f"\n[1] Ingesting METR-LA dataset from {RAW_DIR}...")
    df = load_metr_la(raw_dir=RAW_DIR)
    print(f"    Dataset Shape (timesteps, sensors): {df.shape}")
    print(f"    Index range: {df.index.min()} to {df.index.max()}")
    print(f"    Total sensors: {len(df.columns)}")
    print(f"    Total missing values in raw data: {df.isnull().sum().sum()}")

    # Summary Statistics
    print("\n[2] Summary Statistics (first 5 sensors):")
    print(df.iloc[:, :5].describe().round(2))

    # 2. Time-Series Plots
    print("\n[3] Generating and saving traffic flow visualizations...")
    sensor_id = df.columns[0]
    
    plt.figure(figsize=(12, 5))
    plt.plot(df[sensor_id], color='steelblue', lw=1.2)
    plt.title(f"Traffic Speed Time-Series - Sensor {sensor_id}", fontsize=13)
    plt.xlabel("Timestamp", fontsize=11)
    plt.ylabel("Speed (mph)", fontsize=11)
    plot_path1 = OUTPUTS_DIR / "eda_single_sensor.png"
    plt.tight_layout()
    plt.savefig(plot_path1, dpi=150)
    plt.close()
    print(f"    Saved single-sensor plot to: {plot_path1}")

    plt.figure(figsize=(12, 5))
    for col in df.columns[:5]:
        plt.plot(df[col][:500], label=f"Sensor {col}", alpha=0.8, lw=1.0)
    plt.title("Traffic Flow (First 5 Sensors - 500 Timesteps)", fontsize=13)
    plt.xlabel("Timestamp", fontsize=11)
    plt.ylabel("Speed (mph)", fontsize=11)
    plt.legend(loc="upper right")
    plot_path2 = OUTPUTS_DIR / "eda_multi_sensor.png"
    plt.tight_layout()
    plt.savefig(plot_path2, dpi=150)
    plt.close()
    print(f"    Saved multi-sensor plot to: {plot_path2}")

    # 3. Missing Value Simulation & Imputation
    print("\n" + "=" * 70)
    print("SECTION 3: MISSING DATA SIMULATION & FORWARD-FILL IMPUTATION")
    print("=" * 70)

    print("\n[4] Simulating 30% random sensor dropout...")
    df_missing = simulate_missing_data(df, missing_rate=0.3, random_state=42)
    missing_count = df_missing.isnull().sum().sum()
    total_cells = df_missing.size
    print(f"    Simulated Missing Values: {missing_count} / {total_cells} ({missing_count/total_cells*100:.1f}%)")

    # Plot original vs missing
    plt.figure(figsize=(12, 5))
    sample_steps = 300
    plt.plot(df[sensor_id].values[:sample_steps], label="Original Signal", color='black', lw=1.5)
    plt.plot(df_missing[sensor_id].values[:sample_steps], label="Corrupted (30% Missing)", color='red', alpha=0.7, linestyle='--')
    plt.title(f"Simulated Sensor Dropout Effect (Sensor {sensor_id})", fontsize=13)
    plt.xlabel("Timestep (5-min intervals)", fontsize=11)
    plt.ylabel("Speed (mph)", fontsize=11)
    plt.legend()
    plot_path3 = OUTPUTS_DIR / "missing_data_simulation.png"
    plt.tight_layout()
    plt.savefig(plot_path3, dpi=150)
    plt.close()
    print(f"    Saved missing simulation plot to: {plot_path3}")

    # Forward-fill imputation
    print("\n[5] Applying forward-fill (ffill) imputation...")
    df_filled = df_missing.ffill().bfill()
    remaining_null = df_filled.isnull().sum().sum()
    print(f"    Remaining missing values after imputation: {remaining_null}")
    print("\nDone! Section 1-3 completed successfully.")

if __name__ == "__main__":
    main()
