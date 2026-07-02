# Author: X.M. Christine Zhu

"""
Atracsys FusionTrack interface — ROS subscriber backend.

Architecture
------------
The Atracsys tracker is driven by a SEPARATE process: the `atracsys` ROS node
from sawAtracsysFusionTrack's `ros/` package, typically launched under its
own account, e.g.:

    source /opt/ros/noetic/setup.bash
    source /home/devel/catkin_ws/devel/setup.bash
    rosrun atracsys atracsys -j /path/to/managerMarker_test.json

That node owns the cisst/SAW component in-process (loads the SDK, talks to
the hardware) and bridges every configured tool to a ROS topic via CRTK:

    /atracsys/<body_name>/measured_cp      (geometry_msgs/PoseStamped, metres)

This module does NOT load cisst/SAW components itself — it only subscribes
to those topics. (An earlier version of this file instantiated
mtsAtracsysFusionTrack in-process via the cisst Python bindings directly;
that path deadlocked inside cisst's dynamic ComponentCreate RPC on this
machine and is no longer used. Let the standalone `atracsys` node, which
is known to work, own the hardware.)

Before running any experiment
------------------------------
1. roscore must be running, and the `atracsys` ROS node above must already
   be running and have the bodies you need visible to the camera.

2. The body names passed to get_pose() and collect_samples() must exactly
   match the "name" fields in the managerMarker_test.json that the running
   `atracsys` node was launched with — NOT a path configured here. This
   module has no notion of the JSON config; it only knows ROS topic names.

3. Do NOT set "reference" on any tool in that managerMarker_test.json. When
   "reference" is set, measured_cp() returns the pose relative to the
   reference body (T_AB), not in the tracker frame (T_TB). All experiment
   scripts assume tracker-frame poses and compute relative transforms in
   Python.

Body names used by the experiments in this repo
-------------------------------------------------
    "Anatomy"        — reference rigid body
    "Anspoch_drill"  — drill rigid body

These two bodies cover all three experiments:
    "Anatomy"        — Exp 1 (moved by hand), Exp 2 (world link)
    "Anspoch_drill"  — Exp 1 (stays fixed), Exp 2 (chain end), Exp 3 (attach to Galen EE tip)
"""

import numpy as np

import rospy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation

_TRACKER_NAME  = "AtracsysTracker"
_TOPIC_PREFIX  = "/atracsys"   # matches the `atracsys` node's default ROS node name
_WAIT_TIMEOUT  = 1.0           # seconds to wait for one fresh message before "not visible"


class AtracsysTracker:
    """
    Subscribes to the `atracsys` ROS node's per-body measured_cp topics.

    Usage
    -----
        tracker = AtracsysTracker()
        tracker.connect()

        T = tracker.get_pose("Anatomy")          # 4×4 numpy array
        samples = tracker.collect_samples("Anatomy", n=200)  # (N,4,4)

        tracker.disconnect()
    """

    def __init__(self, topic_prefix: str = _TOPIC_PREFIX,
                 node_name: str = _TRACKER_NAME):
        self._topic_prefix = topic_prefix
        self._node_name    = node_name
        self._owns_node     = False

    # ── connect / disconnect ──────────────────────────────────────────────────

    def connect(self):
        """Initialize a rospy node (if one isn't already running in this process)."""
        if rospy.core.is_initialized():
            print("[connect] rospy node already initialized in this process — reusing it.")
            return
        print("[connect] initializing rospy node...", flush=True)
        rospy.init_node(self._node_name, anonymous=True, disable_signals=True)
        self._owns_node = True
        print("[connect] rospy node ready. Topics are subscribed lazily per body "
              "(first get_pose() call for each body name).")

    def disconnect(self):
        """Shut down the rospy node if this instance created it."""
        if self._owns_node:
            rospy.signal_shutdown("AtracsysTracker.disconnect()")
            self._owns_node = False

    # ── pose acquisition ──────────────────────────────────────────────────────

    def get_pose(self, body_name: str, timeout: float = _WAIT_TIMEOUT) -> np.ndarray:
        """
        Return one 4×4 SE(3) measurement for the named rigid body.

        Blocks for up to `timeout` seconds waiting for the next message on
        `<topic_prefix>/<body_name>/measured_cp`. Raises RuntimeError if no
        message arrives in time (body not visible, or the `atracsys` ROS
        node isn't running / doesn't have this body configured).

        Parameters
        ----------
        body_name : str
            Must match a "name" entry in the running node's managerMarker_test.json.

        Returns
        -------
        T : ndarray, shape (4, 4)
            Pose of the rigid body in the tracker's coordinate frame.
            Translation is in metres (the topic publishes millimetres).
        """
        topic = f"{self._topic_prefix}/{body_name}/measured_cp"
        try:
            msg = rospy.wait_for_message(topic, PoseStamped, timeout=timeout)
        except rospy.ROSException:
            raise RuntimeError(
                f"Body '{body_name}' is not visible (no message on '{topic}' "
                f"within {timeout}s). Check line of sight and marker coverage, "
                f"and that the 'atracsys' ROS node is running with this body "
                f"configured."
            )

        p = msg.pose.position
        q = msg.pose.orientation
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T[:3,  3] = np.array([p.x, p.y, p.z], dtype=np.float64)  # already metres (SI)
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
