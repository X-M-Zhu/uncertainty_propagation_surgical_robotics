# Author: X.M. Christine Zhu

"""
Shared CSV I/O for per-marker position arrays, shape (N, n_markers, 3).

Used by both the collect.py scripts (raw marker_positions samples) and
extract_markers.py (reprojected marker samples), so both sources share one
on-disk layout and can be loaded interchangeably by the analyze_markers.py
scripts.
"""

import pathlib
import numpy as np


def save_marker_csv(path: pathlib.Path, markers: np.ndarray) -> None:
    """Save (N, n_markers, 3) as a flat CSV: N rows x (n_markers*3) columns."""
    N, n_markers, _ = markers.shape
    flat = markers.reshape(N, n_markers * 3)
    header = ",".join(
        f"m{k}_{ax}" for k in range(n_markers) for ax in ("x", "y", "z")
    )
    np.savetxt(str(path), flat, delimiter=",", header=header, comments="")


def load_marker_csv(path: pathlib.Path) -> np.ndarray:
    """Load a flattened marker CSV back into (N, n_markers, 3)."""
    data = np.loadtxt(str(path), delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    n_markers = data.shape[1] // 3
    return data.reshape(-1, n_markers, 3)
