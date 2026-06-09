# Author: X.M. Christine Zhu

"""
Atracsys FusionTrack interface — cisst/SAW backend.

Uses cisst/sawAtracsysFusionTrack framework, the same stack
used throughout the JHU surgical robotics lab.

Before running any experiment
------------------------------
1. Set CONFIG_PATH below to the full path of your
   configAtracsysFusionTrack.json file.

2. For each rigid body you want to track, make sure it has:
     a) A geometry JSON file (see hardware/atracsys/atracsys/geometry_*.json
        for the format — four fiducial XYZ positions in mm, an integer id,
        and an optional pivot point).
     b) An entry in managerMarker.json under "tools".

3. The body names passed to get_pose() and collect_samples() must exactly
   match the "name" fields in managerMarker.json.

Body names currently in managerMarker.json
------------------------------------------
    "Anatomy"        — reference rigid body (4 markers, id=1)
    "Anspoch_drill"  — tool rigid body, expressed relative to "Anatomy"

These two bodies cover all three experiments:
    "Anatomy"        — Exp 1 (move by hand), Exp 2 (world link), Exp 3 (keep visible as reference)
    "Anspoch_drill"  — Exp 2 (relative link), Exp 3 (attach to Galen EE tip)
"""

import os
import time
import numpy as np

# ── Path to your tracker configuration file ───────────────────────────────────
# Update this to the actual path on your machine.
CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "hardware", "atracsys", "atracsys", "core", "share",
    "configAtracsysFusionTrack.json"
)

_TRACKER_NAME = "AtracsysTracker"


class AtracsysTracker:
    """
    Wrapper around sawAtracsysFusionTrack (cisst/SAW).

    Usage
    -----
        tracker = AtracsysTracker()
        tracker.connect()

        T = tracker.get_pose("Anatomy")          # 4×4 numpy array
        samples = tracker.collect_samples("Anatomy", n=200)  # (N,4,4)

        tracker.disconnect()
    """

    def __init__(self, config_path: str = CONFIG_PATH,
                 tracker_name: str = _TRACKER_NAME):
        self._config_path  = config_path
        self._tracker_name = tracker_name
        self._manager      = None
        self._proxy        = None
        self._interfaces   = {}   # body_name -> cisst interface

    # ── connect / disconnect ──────────────────────────────────────────────────

    def connect(self):
        """Initialize cisst/SAW, load the Atracsys component, and start it."""
        import cisstCommonPython as cmn
        import cisstMultiTaskPython as mts

        # Suppress verbose cisst logging — comment out to debug hardware issues
        cmn.cmnLogger_SetMask(cmn.CMN_LOG_ALLOW_ERRORS_AND_WARNINGS)
        cmn.cmnLogger_SetMaskFunction(cmn.CMN_LOG_ALLOW_ERRORS_AND_WARNINGS)
        cmn.cmnLogger_SetMaskDefaultLog(cmn.CMN_LOG_ALLOW_ERRORS_AND_WARNINGS)
        cmn.cmnLogger_SetMaskClassMatching("mts", cmn.CMN_LOG_ALLOW_ERRORS_AND_WARNINGS)

        self._manager = mts.mtsManagerLocal.GetInstance()
        self._manager.CreateAllAndWait(5.0)
        self._manager.StartAllAndWait(5.0)

        self._proxy = mts.mtsComponentWithManagement(
            f"{self._tracker_name}Proxy"
        )
        self._manager.AddComponent(self._proxy)
        self._proxy.CreateAndWait(5.0)
        time.sleep(0.5)

        services = self._proxy.GetManagerComponentServices()

        result = services.Load("sawAtracsysFusionTrack")
        if not result:
            raise RuntimeError("Failed to load sawAtracsysFusionTrack. "
                               "Is the cisst/SAW library installed and on your path?")

        import cisstMultiTaskPython as mts_inner
        args = mts_inner.mtsTaskContinuousConstructorArg(self._tracker_name)
        result = services.ComponentCreate("mtsAtracsysFusionTrack", args)
        if not result:
            raise RuntimeError(f"Failed to create mtsAtracsysFusionTrack component.")

        component = self._manager.GetComponent(self._tracker_name)
        component.Configure(self._config_path)
        component.CreateAndWait(5.0)
        component.StartAndWait(5.0)

        # Give the tracker time to initialize and start streaming frames
        print(f"Tracker component started. Waiting for initialization...")
        time.sleep(4.5)
        print("Tracker ready.")

    def disconnect(self):
        """Stop and clean up the cisst component manager."""
        if self._manager is not None:
            try:
                self._manager.KillAllAndWait(5.0)
                self._manager.Cleanup()
            except Exception:
                pass
            self._manager = None
        self._interfaces.clear()

    # ── pose acquisition ──────────────────────────────────────────────────────

    def _get_interface(self, body_name: str):
        """Return (or lazily create) the cisst interface for body_name."""
        if body_name not in self._interfaces:
            iface = self._proxy.AddInterfaceRequiredAndConnect(
                (self._tracker_name, body_name)
            )
            if iface is None:
                raise RuntimeError(
                    f"Could not connect to body '{body_name}'. "
                    f"Check that it is listed in managerMarker.json and "
                    f"its geometry JSON file is present."
                )
            self._interfaces[body_name] = iface
        return self._interfaces[body_name]

    def get_pose(self, body_name: str) -> np.ndarray:
        """
        Return one 4×4 SE(3) measurement for the named rigid body.

        Raises RuntimeError if the body is not currently visible.

        Parameters
        ----------
        body_name : str
            Must match a "name" entry in managerMarker.json exactly.

        Returns
        -------
        T : ndarray, shape (4, 4)
            Pose of the rigid body in the tracker's coordinate frame.
            Translation is in metres.
        """
        iface = self._get_interface(body_name)
        pose  = iface.measured_cp()

        if not pose.GetValid():
            raise RuntimeError(
                f"Body '{body_name}' is not visible. "
                "Check line of sight and marker coverage."
            )

        # Convert cisst vctFrm3 → 4×4 numpy array
        frm3 = pose.Position()
        T = np.eye(4, dtype=float)
        T[:3, :3] = np.array(frm3.Rotation(),    dtype=float)
        T[:3,  3] = np.array(frm3.Translation(), dtype=float) * 1e-3  # mm → m
        return T

    def collect_samples(self, body_name: str, n: int = 200,
                        verbose: bool = True) -> np.ndarray:
        """
        Collect n consecutive measurements for body_name.

        Skips frames where the body is not visible and retries until
        n valid measurements are collected (up to 3× attempts).

        Returns
        -------
        samples : ndarray, shape (n, 4, 4)
        """
        samples  = []
        attempts = 0
        max_attempts = n * 3

        while len(samples) < n and attempts < max_attempts:
            attempts += 1
            try:
                samples.append(self.get_pose(body_name))
            except RuntimeError:
                pass   # body temporarily occluded — retry

            if verbose and len(samples) % 50 == 0 and len(samples) > 0:
                print(f"  {len(samples)}/{n} samples collected")

        if len(samples) < n:
            raise RuntimeError(
                f"Only collected {len(samples)}/{n} valid samples for "
                f"'{body_name}' after {attempts} attempts. "
                "Check line of sight."
            )

        return np.stack(samples[:n], axis=0)
