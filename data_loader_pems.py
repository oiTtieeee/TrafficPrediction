"""
Data loading utilities for PEMS-BAY traffic speed dataset and road network graph files.
PEMS-BAY: 325 loop detector sensors, San Francisco Bay Area, CA.
          6 months of 5-minute interval speed data (Jan 1 - May 31, 2017).
          52,116 timesteps x 325 sensors.
"""
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import h5py
except ImportError:
    h5py = None

# Download URLs (Zenodo record 4263971 = DCRNN benchmark datasets)
PEMS_BAY_H5_URLS = [
    "https://zenodo.org/record/4263971/files/pems-bay.h5",
    "https://zenodo.org/records/4263971/files/pems-bay.h5",
]

DCRNN_PEMS_BASE = "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/"
PEMS_BAY_GRAPH_FILES = ["distances_bay_2017.csv", "graph_sensor_ids_bay.txt"]


def load_pems_bay(raw_dir: Path) -> pd.DataFrame:
    """
    Loads the PEMS-BAY speed dataset into a pandas DataFrame.
    Checks local cache, extracts from zip if found, or downloads from URL.

    Returns
    -------
    df : pd.DataFrame
        Shape (T, 325) -- timestamps x sensor columns.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    h5_file = raw_dir / "pems-bay.h5"
    zip_file = raw_dir / "pems-bay.h5.zip"

    if not h5_file.exists():
        if zip_file.exists():
            print(f"Extracting {zip_file} ...")
            with zipfile.ZipFile(zip_file, "r") as zf:
                zf.extractall(raw_dir)
            extracted = list(raw_dir.glob("pems-bay.h5"))
            if extracted:
                h5_file = extracted[0]
        else:
            for url in PEMS_BAY_H5_URLS:
                try:
                    print(f"Downloading PEMS-BAY from {url} ...")
                    urllib.request.urlretrieve(url, h5_file)
                    if h5_file.stat().st_size > 1_000_000:
                        break
                    h5_file.unlink()
                except Exception as exc:
                    print(f"  Download failed ({exc}), trying fallback ...")

    if not h5_file.exists():
        raise FileNotFoundError(
            f"PEMS-BAY HDF5 file not found at {h5_file}. "
            f"Please download pems-bay.h5 manually and place it in {raw_dir}."
        )

    # h5py fallback — handle both metr-la ('df') and pems-bay ('speed') layouts
    if h5py is None:
        raise ImportError("h5py is required as a fallback HDF5 reader.")
    with h5py.File(h5_file, "r") as f:
        # Determine the group name
        group_name = "speed" if "speed" in f else "df"
        grp = f[group_name]
        raw_cols = grp["axis0"][:]
        # Sensor IDs may be int64 or bytes
        cols = [c.decode() if isinstance(c, bytes) else str(c) for c in raw_cols]
        raw_idx = grp["axis1"][:]
        idx = pd.to_datetime(raw_idx)
        vals = grp["block0_values"][:]  # shape (T, N)
        df = pd.DataFrame(vals, index=idx, columns=cols)

    return df


def download_pems_bay_graph(raw_dir: Path) -> dict:
    """
    Downloads PEMS-BAY graph files (distances CSV + sensor ID list) from DCRNN repo.

    Returns
    -------
    paths : dict[str, Path]
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for fname in PEMS_BAY_GRAPH_FILES:
        target = raw_dir / fname
        if not target.exists():
            url = DCRNN_PEMS_BASE + fname
            print(f"Downloading {fname} ...")
            try:
                urllib.request.urlretrieve(url, target)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not download {fname}: {exc}\n"
                    f"Please download manually from: {url}\n"
                    f"and place it in {raw_dir}."
                )
        paths[fname] = target
    return paths


def build_pems_bay_adjacency(raw_dir: Path, epsilon: float = 0.1):
    """
    Constructs the normalised Gaussian-kernel adjacency matrix for PEMS-BAY.
    Mirrors the METR-LA graph_utils.build_adjacency procedure exactly.

    Returns
    -------
    adj_mx : np.ndarray, shape (325, 325)
    sensor_ids : list[str]
    """
    paths = download_pems_bay_graph(raw_dir)
    dist_df = pd.read_csv(paths["distances_bay_2017.csv"],
                          header=None, names=["from", "to", "cost"],
                          dtype={"from": str, "to": str})
    with open(paths["graph_sensor_ids_bay.txt"]) as f:
        sensor_ids = f.read().strip().split(",")

    n = len(sensor_ids)
    id2idx = {sid: i for i, sid in enumerate(sensor_ids)}
    adj_mx = np.zeros((n, n), dtype=np.float32)

    distances = dist_df["cost"].values
    sigma2 = float(np.var(distances))
    kappa = float(np.std(distances)) * epsilon

    for _, row in dist_df.iterrows():
        i = id2idx.get(str(row["from"]))
        j = id2idx.get(str(row["to"]))
        if i is None or j is None:
            continue
        d = float(row["cost"])
        if d <= kappa:
            adj_mx[i, j] = float(np.exp(-(d ** 2) / sigma2))

    np.fill_diagonal(adj_mx, 1.0)
    return adj_mx, sensor_ids
