# Robust Spatiotemporal Traffic Flow Forecasting Under Historical Data Scarcity and Sensor Faults: A Residual Graph Neural Network Approach

Official repository and research artifact package for the research paper:  
**"Robust Spatiotemporal Traffic Flow Forecasting Under Historical Data Scarcity and Sensor Faults: A Residual Graph Neural Network Approach"**  
*Target Venues: IEEE Transactions on Intelligent Transportation Systems / IEEE Access*

---

## 📌 Abstract
Accurate, real-time traffic speed forecasting is foundational to proactive Intelligent Transportation Systems (ITS). While Spatiotemporal Graph Neural Networks (GNNs) achieve superior performance on dense historical benchmarks, two critical deployment bottlenecks remain largely unaddressed: **(1) historical data scarcity during newly commissioned corridor deployments (the cold-start problem)**, and **(2) severe temporal lag and over-smoothing during rapid congestion shockwaves**.

In this work, we propose **TrafficGCN-Delta**, a fault-tolerant spatiotemporal framework coupling localized spectral graph convolutions with a residual velocity change formulation ($\Delta y_t = y_t - x_{t-1}$). Evaluated on the benchmark METR-LA highway network (207 loop detectors, 34,272 timesteps) across three data availability regimes (100%, 50%, and 20% history) and simulated 30% sensor dropouts, **TrafficGCN-Delta retains an MAE of 3.61 mph at only 20% data availability—an error degradation of merely 4.9%**, whereas classical models degrade by over 13.5% to 15.5%.

![Architecture](paper/figures/fig1_system_architecture.png)

---

## 📂 Repository Structure

```
├── main.py                                 # Master CLI pipeline runner
├── requirements.txt                        # Pinned dependencies
├── README.md                               # Repository documentation
│
├── paper/                                  # Complete research paper package
│   ├── main.tex                            # IEEE Transactions double-column LaTeX manuscript
│   ├── manuscript.md                       # Full formatted manuscript (academic Markdown)
│   ├── references.bib                      # Curated BibTeX bibliography
│   ├── cover_letter.tex / .md              # Formal Cover Letter to the Editor-in-Chief
│   ├── highlights.md                       # Research highlights
│   └── figures/                            # 5 Publication figures (300 DPI)
│       ├── fig1_system_architecture.png
│       ├── fig2_sensor_network_map.png
│       ├── fig3_data_scarcity_curves.png
│       ├── fig4_residual_delta_analysis.png
│       └── fig5_sensor_dropout_effect.png
│
├── scripts/                                # Standalone executable Python scripts per section
│   ├── 01_eda_and_missing_data.py          # Section 1-3: Ingestion, EDA & 30% dropout simulation
│   ├── 02_single_sensor_baselines.py       # Section 4-5: Autoregressive Linear Regression & LSTM
│   ├── 03_sensor_graph_topology.py         # Section 6-7: Distance matrix, Gaussian adjacency & map
│   ├── 04_multivariate_baselines.py        # Section 8: Network-wide Linear Regression & Random Forest
│   ├── 05_gcn_traffic_forecasting.py       # Section 9-11: Leak-free scaling, TrafficGCN & Delta GCN
│   ├── 06_robustness_and_benchmarks.py     # Section 12-14: 100%/50%/20% regimes & benchmark export
│   └── generate_paper_figures.py           # Generates all 5 figures at 300 DPI
│
├── src/                                    # Reusable Python package
│   ├── config.py                           # Project paths & constants (SEQ_LENGTH = 12)
│   ├── data_loader.py                      # Dataset loader (HDF5) & graph file ingestion
│   ├── graph_utils.py                      # Adjacency matrix construction & PyG edge tensors
│   ├── preprocessing.py                    # Sequence generation & leak-free standardization
│   ├── models/                             # Baseline & GCN architectures
│   ├── train.py                            # Mini-batch collation & training loops
│   └── evaluate.py                         # Evaluation metrics (MAE, RMSE) & CSV logging
│
├── notebooks/                              # Phase-separated Jupyter notebooks (01 to 06)
├── data/raw/                               # METR-LA speed dataset & road graph CSVs
├── models/checkpoints/                     # Trained model weights (.pt)
└── outputs/                                # final_results.csv and visualization outputs
```

---

## 🚀 Quick Start & Reproduction

### 1. Environment Setup
```bash
git clone https://github.com/YourUsername/TrafficGCN-Delta.git
cd TrafficGCN-Delta
pip install -r requirements.txt
```

### 2. Run Individual Sections via Master CLI
```bash
# Step 1: Ingest METR-LA dataset & run EDA
python main.py --step 1

# Step 2: Single-Sensor Baselines (Linear Regression & LSTM)
python main.py --step 2

# Step 3: Compute Adjacency Matrix & Plot Network Topology
python main.py --step 3

# Step 4: Multivariate Baselines across all 207 sensors
python main.py --step 4

# Step 5: Spatiotemporal GCN & Residual Momentum Modeling
python main.py --step 5

# Step 6: Data Scarcity Experiments (100%, 50%, 20%) & CSV Export
python main.py --step 6
```

### 3. Run Entire End-to-End Pipeline
```bash
python main.py --all
```

### 4. Regenerate All Paper Figures (300 DPI)
```bash
python scripts/generate_paper_figures.py
```

---

## 📊 Benchmark Results

Performance comparison across data availability regimes on the METR-LA highway network:

| Model Architecture | 100% Data MAE | 100% Data RMSE | 50% Data MAE | 50% Data RMSE | 20% Data MAE | 20% Data RMSE | Max MAE Degradation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (Multivariate)** | 4.12 mph | 8.25 mph | 4.39 mph | 8.64 mph | 4.76 mph | 9.10 mph | +15.53% |
| **Random Forest (Multivariate)** | 3.86 mph | 7.82 mph | 4.05 mph | 8.12 mph | 4.38 mph | 8.55 mph | +13.47% |
| **Direct TrafficGCN** | 3.65 mph | 7.34 mph | 3.78 mph | 7.55 mph | 3.92 mph | 7.81 mph | +7.39% |
| **TrafficGCN-Delta (Proposed)** | **3.44 mph** | **6.95 mph** | **3.52 mph** | **7.12 mph** | **3.61 mph** | **7.28 mph** | **+4.94%** |

---

## 📖 Citation

If you find this codebase or paper useful in your research, please cite:

```bibtex
@article{sharma2026robust,
  author    = {Atharva Sharma and Aditya Thakur and Kavish Jha and Pranjal Patidar and Pankaj Malik},
  title     = {Robust Spatiotemporal Traffic Flow Forecasting Under Historical Data Scarcity and Sensor Faults: A Residual Graph Neural Network Approach},
  journal   = {IEEE Transactions on Intelligent Transportation Systems},
  year      = {2026}
}
```
