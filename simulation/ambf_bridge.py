"""
AMBF Bridge — runs INSIDE WSL.

Connects to a running AMBF simulator via ambf_client (ROS-based),
reads joint states for selected robots, and streams them as
newline-delimited JSON to stdout so simulate.py (Windows side)
can read them through a subprocess pipe.

Launch from Windows like:
    wsl bash -lc "python3 /mnt/c/.../simulation/ambf_bridge.py PSM ECM"

Requires:
    - ROS sourced in the bash login profile (~/.bashrc or ~/.profile)
    - ambf_client installed in the ROS workspace
    - AMBF simulator already running
"""

import sys
import json
import time

# ── AMBF namespace map (matches node_registry.py) ─────────────────────────────
# Keys are GUI robot names; values are the AMBF object handle names.
# Adjust if your AMBF scene uses different names.
HANDLE_NAMES = {
    "PSM":    "psm/baselink",
    "ECM":    "ecm/baselink",
    "MTM":    "mtm/TopPanel",
    "Raven2": "raven2/baselink",
}

POLL_HZ = 30   # update rate sent to Windows side


def main(robot_names):
    try:
        from ambf_client import Client
    except ImportError:
        sys.stderr.write("[ambf_bridge] ERROR: ambf_client not found. "
                         "Make sure ROS is sourced and ambf_client is installed.\n")
        sys.exit(1)

    client = Client(client_name="unc_bridge")
    try:
        client.connect(default_publish_rate=120)
    except Exception as e:
        sys.stderr.write(f"[ambf_bridge] ERROR: could not connect to AMBF: {e}\n")
        sys.exit(1)

    # Give ROS a moment to discover topics
    time.sleep(1.5)

    handles = {}
    for name in robot_names:
        key = HANDLE_NAMES.get(name)
        if key is None:
            sys.stderr.write(f"[ambf_bridge] WARNING: no handle mapping for '{name}'\n")
            continue
        h = client.get_obj_handle(key)
        if h is None:
            sys.stderr.write(f"[ambf_bridge] WARNING: AMBF object '{key}' not found "
                             f"(is it loaded in the scene?)\n")
        else:
            handles[name] = h
            sys.stderr.write(f"[ambf_bridge] Connected to '{key}' for robot '{name}'\n")

    if not handles:
        sys.stderr.write("[ambf_bridge] ERROR: no robot handles found — exiting.\n")
        sys.exit(1)

    interval = 1.0 / POLL_HZ
    sys.stderr.write(f"[ambf_bridge] Streaming {list(handles.keys())} at {POLL_HZ} Hz\n")

    while True:
        t0 = time.time()
        states = {}
        for name, h in handles.items():
            try:
                joints = h.get_all_joint_pos()
                if joints:
                    states[name] = list(joints)
            except Exception:
                pass

        # Emit one JSON line — simulate.py reads this with readline()
        print(json.dumps(states), flush=True)

        elapsed = time.time() - t0
        sleep = interval - elapsed
        if sleep > 0:
            time.sleep(sleep)


if __name__ == "__main__":
    # Accept optional robot name list as CLI args; default to all known robots
    names = sys.argv[1:] if len(sys.argv) > 1 else list(HANDLE_NAMES.keys())
    main(names)
