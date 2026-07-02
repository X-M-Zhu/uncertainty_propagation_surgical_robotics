# Author: X.M. Christine Zhu

"""
Experiment 1 — Marker extraction: derive raw fiducial positions per sample.

Per your mentor's request, this derives EACH marker's tracker-frame position
(not just the fitted rigid-body pose) for every sample already collected by
collect.py. No new tracker queries are needed: the registered pose T already
encodes the rigid-body fit, and each marker's LOCAL position (in the tool's
own frame) is fixed and stored in its geometry JSON file. For each sample:

    p_marker_tracker = T[:3, :3] @ p_marker_local + T[:3, 3]

Caveat
------
This is a *reprojection*, not an independent raw measurement: it assumes the
tool is perfectly rigid (local marker geometry never changes) and reuses the
already-fitted pose T. It does NOT recover marker-level noise that the
Atracsys SDK's internal rigid-body fit may have already averaged out — true
raw fiducial correspondences are not exposed by this driver (see
atracsys_interface.py docstring). What it DOES give you: each marker's actual
3-D position in the tracker frame at each sample, useful for visualizing
marker spread / sanity-checking the rigid-body assumption.

Reads
-----
  data_fixed_drill/<position>/bodyA.csv   (drill frame samples,   from collect.py)
  data_fixed_drill/<position>/bodyB.csv   (Anatomy frame samples, from collect.py)
  hardware/atracsys/atracsys/geometry_anspoch_drill.json
  hardware/atracsys/atracsys/geometry_anatomy_reference_5_24.json

Writes
------
  data_fixed_drill/<position>/markersA.csv   (drill markers,   N_samples x (n_markers*3))
  data_fixed_drill/<position>/markersB.csv   (Anatomy markers, N_samples x (n_markers*3))
"""

import sys
import json
import pathlib
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_EXP  = _HERE.parent
_ROOT = _EXP.parent
sys.path.insert(0, str(_EXP))

from utils.se3_stats import load_poses_csv

DATA_DIR = _HERE / "data_fixed_drill"

GEOMETRY_DRILL   = _ROOT / "hardware" / "atracsys" / "atracsys" / "geometry_anspoch_drill.json"
GEOMETRY_ANATOMY = _ROOT / "hardware" / "atracsys" / "atracsys" / "geometry_anatomy_reference_5_24.json"


def load_fiducials_m(geometry_path: pathlib.Path) -> np.ndarray:
    """
    Load a geometry JSON's fiducial local positions, converted mm -> m.

    Returns
    -------
    ndarray, shape (n_markers, 3)
    """
    with open(geometry_path) as f:
        geom = json.load(f)
    pts = np.array(
        [[fid["x"], fid["y"], fid["z"]] for fid in geom["fiducials"]],
        dtype=np.float64,
    ) * 1e-3   # mm -> m
    return pts


def derive_marker_positions(frame_samples: np.ndarray, fiducials_local: np.ndarray) -> np.ndarray:
    """
    Reproject each local fiducial through every sampled frame.

    Parameters
    ----------
    frame_samples : ndarray, shape (N, 4, 4)
    fiducials_local : ndarray, shape (n_markers, 3)

    Returns
    -------
    ndarray, shape (N, n_markers, 3)
        Marker positions in the tracker frame, per sample.
    """
    N = frame_samples.shape[0]
    n_markers = fiducials_local.shape[0]
    out = np.zeros((N, n_markers, 3), dtype=np.float64)
    for i in range(N):
        R = frame_samples[i, :3, :3]
        t = frame_samples[i, :3, 3]
        out[i] = fiducials_local @ R.T + t
    return out


def save_marker_csv(path: pathlib.Path, markers: np.ndarray) -> None:
    """Save (N, n_markers, 3) as a flat CSV: N rows x (n_markers*3) columns."""
    N, n_markers, _ = markers.shape
    flat = markers.reshape(N, n_markers * 3)
    header = ",".join(
        f"m{k}_{ax}" for k in range(n_markers) for ax in ("x", "y", "z")
    )
    np.savetxt(str(path), flat, delimiter=",", header=header, comments="")


def main():
    if not GEOMETRY_DRILL.exists() or not GEOMETRY_ANATOMY.exists():
        raise FileNotFoundError(
            f"Geometry files not found. Expected:\n  {GEOMETRY_DRILL}\n  {GEOMETRY_ANATOMY}"
        )
    fid_drill   = load_fiducials_m(GEOMETRY_DRILL)
    fid_anatomy = load_fiducials_m(GEOMETRY_ANATOMY)
    print(f"Loaded {len(fid_drill)} drill markers, {len(fid_anatomy)} Anatomy markers.")

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Run collect.py first. Expected: {DATA_DIR}")

    pos_dirs = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())
    if not pos_dirs:
        raise FileNotFoundError(f"No position folders found in {DATA_DIR}")

    for pos_dir in pos_dirs:
        bodyA_path = pos_dir / "bodyA.csv"
        bodyB_path = pos_dir / "bodyB.csv"
        if not (bodyA_path.exists() and bodyB_path.exists()):
            print(f"  {pos_dir.name}: missing bodyA/bodyB.csv, skipping")
            continue

        samples_A = load_poses_csv(str(bodyA_path))
        samples_B = load_poses_csv(str(bodyB_path))

        markers_A = derive_marker_positions(samples_A, fid_drill)
        markers_B = derive_marker_positions(samples_B, fid_anatomy)

        save_marker_csv(pos_dir / "markersA.csv", markers_A)
        save_marker_csv(pos_dir / "markersB.csv", markers_B)
        print(f"  {pos_dir.name}: wrote markersA.csv ({markers_A.shape}), "
              f"markersB.csv ({markers_B.shape})")

    print("\nDone. Run analyze_markers.py to see per-marker statistics.")


if __name__ == "__main__":
    main()
