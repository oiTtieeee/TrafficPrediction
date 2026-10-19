"""
Data loading utilities for METR-LA traffic speed dataset and road network graph files.
"""
import os
import zipfile
import urllib.request
import pandas as pd
from pathlib import Path
from src.config import RAW_DIR

try:
    import h5py
except ImportError:
    h5py = None

DCRNN_GRAPH_BASE = "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/"
GRAPH_FILES = ["distances_la_2012.csv", "graph_sensor_ids.txt", "graph_sensor_locations.csv"]

METR_LA_URL = "https://raw.githubusercontent.com/leilin-research/GCGRNN/master/data/METR-LA_traffic_speed/metr-la.h5"

def load_metr_la(raw_dir=None):
    """
    Loads the METR-LA speed dataset into a pandas DataFrame.
    Checks local directories and extracts from zip if necessary.
    """
    raw_dir = Path(raw_dir or RAW_DIR)
    h5_file = raw_dir / "metr-la.h5"
    zip_file = raw_dir / "METR-LA.h5.zip"

    # Also check project root for convenience
    root_h5 = Path("METR-LA.h5")
    root_zip = Path("METR-LA.h5.zip")

    if not h5_file.exists():
        if root_h5.exists():
            h5_file = root_h5
        elif zip_file.exists():
            print(f"Extracting {zip_file}...")
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(raw_dir)
            extracted = list(raw_dir.glob("*.h5"))
            if extracted:
                h5_file = extracted[0]
        elif root_zip.exists():
            print(f"Extracting {root_zip}...")
            with zipfile.ZipFile(root_zip, 'r') as zf:
                zf.extractall(raw_dir)
            extracted = list(raw_dir.glob("*.h5"))
            if extracted:
                h5_file = extracted[0]
        else:
            print(f"Downloading METR-LA dataset to {h5_file}...")
            urllib.request.urlretrieve(METR_LA_URL, h5_file)

    try:
        # Standard pandas read_hdf
        df = pd.read_hdf(h5_file)
    except Exception:
        # Direct h5py fallback
        with h5py.File(h5_file, 'r') as f:
            cols = [c.decode() for c in f['df/axis0'][:]]
            idx = pd.to_datetime(f['df/axis1'][:])
            values_raw = f['df/block0_values'][:]
            df = pd.DataFrame(values_raw, index=idx, columns=cols)

    return df

def download_graph_files(raw_dir=None):
    """
    Downloads and caches distances, sensor IDs, and GPS locations if not present.
    """
    raw_dir = Path(raw_dir or RAW_DIR)
    paths = {}
    for fname in GRAPH_FILES:
        target = raw_dir / fname
        root_target = Path(fname)
        if not target.exists():
            if root_target.exists():
                target = root_target
            else:
                print(f"Downloading {fname}...")
                urllib.request.urlretrieve(DCRNN_GRAPH_BASE + fname, target)
        paths[fname] = target
    return paths
