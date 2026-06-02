# Simulation-Based Uncertainty Propagation in Geometric Networks for Surgical Robotics

**Author:** X.M. Christine Zhu &nbsp;·&nbsp; **Mentor:** Dr. Russell H. Taylor

---

This repository implements and validates a mathematical framework for **uncertainty propagation in geometric networks**, following the CIS I right-multiplicative perturbation convention.

Built for surgical robotics applications where multiple sensors, rigid links, and coordinate
frames form a network, and you need to know how measurement errors travel through that network
to affect a final quantity (e.g. tool-tip position, distance between two anatomical landmarks).

The math is documented in `docs/` and `PSEUDOCODE.md`.

---

## Installation

### For CIS I students
> **Requires Python 3.10 or later.** No need to download or clone anything.

**1. Install**
```bash
pip install "uncertainty-networks[gui] @ git+https://github.com/X-M-Zhu/uncertainty_propagation_surgical_robotics.git"
```

**2. Launch the GUI**
```bash
uncertainty-gui
```

That's it. The visualiser will open.  
For the full API reference see [API_REFERENCE.md](API_REFERENCE.md).

### For developers

```bash
git clone https://github.com/X-M-Zhu/uncertainty_propagation_surgical_robotics.git
cd uncertainty_propagation_surgical_robotics
pip install -e ".[gui]"
```

Verify the install:

```bash
pytest
```

All tests should pass.

### For AMBF live simulation

The GUI has two modes:

- **Mock mode** (default) — joint angles driven by sine waves. No AMBF or ROS needed. Works on any platform.
- **Live mode** — joint angles streamed from a running AMBF simulator via `simulation/ambf_bridge.py`. Requires AMBF + ROS in a Linux environment.

### For Linux users

**1. Install ROS Noetic**

```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" \
    > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' \
    --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install -y ros-noetic-desktop-full python3-rosdep python3-catkin-tools
sudo rosdep init && rosdep update
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

**2. Clone and build AMBF**

```bash
cd ~
git clone https://github.com/WPI-AIM/ambf.git --recurse-submodules
cd ambf && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**3. Install ambf_client**

```bash
cd ~/ambf/ambf_ros_modules/ambf_client
pip3 install .
```

**4. Run AMBF**

```bash
cd ~/ambf/build
./ambf_simulator --launch_file ../ambf/launch.yaml -l 0,1,2
```

**5. Launch the GUI in Live mode**

With AMBF running, you can start the bridge manually in a second terminal:

```bash
source /opt/ros/noetic/setup.bash
python3 simulation/ambf_bridge.py PSM ECM
```

Or simply launch the GUI, select **Live (AMBF)** mode — it starts the bridge automatically.

---

### For Windows users (via WSL2)

`simulate.py` spawns the bridge inside WSL2 from Windows via `wsl bash -lc "..."`.
AMBF itself runs in WSL2; the GUI runs natively on Windows.

**1. Install WSL2 with Ubuntu 20.04**

Open PowerShell as Administrator:

```powershell
wsl --install -d Ubuntu-20.04
```

Restart when prompted. Open **Ubuntu 20.04** from the Start menu and set a username and password.

**2. Install ROS Noetic inside WSL2**

In the Ubuntu WSL terminal:

```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" \
    > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' \
    --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install -y ros-noetic-desktop-full python3-rosdep python3-catkin-tools
sudo rosdep init && rosdep update
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
```

**3. Clone and build AMBF inside WSL2**

```bash
cd ~
git clone https://github.com/WPI-AIM/ambf.git --recurse-submodules
cd ambf && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**4. Install ambf_client inside WSL2**

```bash
cd ~/ambf/ambf_ros_modules/ambf_client
pip3 install .
```

**5. Run AMBF inside WSL2**

- **Windows 11** — WSLg provides a display automatically:
  ```bash
  source /opt/ros/noetic/setup.bash
  cd ~/ambf/build
  ./ambf_simulator --launch_file ../ambf/launch.yaml -l 0,1,2
  ```

- **Windows 10** — install [VcXsrv](https://sourceforge.net/projects/vcxsrv/), launch it (check *Disable access control*), then in WSL:
  ```bash
  export DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0
  source /opt/ros/noetic/setup.bash
  cd ~/ambf/build
  ./ambf_simulator --launch_file ../ambf/launch.yaml -l 0,1,2
  ```

**6. Launch the GUI on Windows**

```bash
pip install "uncertainty-networks[gui]"
uncertainty-gui
```

In the GUI's **Live (AMBF)** settings, set the **ROS source command** to:

```
source /opt/ros/noetic/setup.bash
```

Select robots and click **Live (AMBF)**. The GUI will automatically call:

```
wsl bash -lc "source /opt/ros/noetic/setup.bash && python3 /mnt/c/.../ambf_bridge.py PSM ECM"
```

---

### For macOS users

AMBF does not have a native macOS build. The recommended approach is **VMware Fusion** — a virtual machine app that creates a full Linux computer running inside your Mac. Think of it as a second computer living in a window on your screen. AMBF, ROS, and the GUI all run inside that Linux window, exactly as they do on a real Linux machine.

**Step 1 — Download VMware Fusion**

Go to [https://www.vmware.com/products/fusion.html](https://www.vmware.com/products/fusion.html) and download VMware Fusion. It is free for personal use. Open the downloaded `.dmg` file and drag VMware Fusion into your Applications folder to install it.

**Step 2 — Download the Ubuntu 20.04 ISO**

An ISO file is a disk image — it contains everything needed to install Ubuntu Linux. Download it from:
[https://releases.ubuntu.com/20.04/](https://releases.ubuntu.com/20.04/)

Click the link named **64-bit PC (AMD64) desktop image**. The file will be named something like `ubuntu-20.04.6-desktop-amd64.iso` and is about 3 GB. Wait for the download to finish before continuing.

**Step 3 — Create a new virtual machine**

1. Open VMware Fusion from your Applications folder.
2. Click **File → New** (or the **+** button).
3. Drag the downloaded `.iso` file into the window that appears, then click **Continue**.
4. Select **Linux → Ubuntu 64-bit** if prompted for the operating system type, then click **Continue**.
5. On the **Finish** screen, click **Customize Settings** before clicking Finish.
   - Set **Processors & Memory** to at least **4 CPU cores** and **8 GB RAM**.
   - Set **Hard Disk** to at least **40 GB**.
6. Close the settings window, then click **Finish**. The VM will start.

**Step 4 — Install Ubuntu inside the VM**

A purple Ubuntu installer screen will appear inside the VMware window.

1. Click **Install Ubuntu**.
2. Choose your keyboard layout and click **Continue**.
3. Select **Normal installation** and click **Continue**.
4. Select **Erase disk and install Ubuntu** (this only erases the virtual disk, not your Mac), then click **Install Now → Continue**.
5. Choose your time zone and click **Continue**.
6. Enter your name, a computer name, a username, and a password. Click **Continue**.
7. Wait 10–15 minutes for installation to complete. Click **Restart Now** when prompted.
8. Press Enter when asked to remove the installation medium (VMware handles this automatically).

Ubuntu will boot to a desktop inside the VMware window. Log in with the password you set.

**Step 5 — Open a Terminal in Ubuntu**

Press **Ctrl + Alt + T** on your keyboard. A terminal window will open — this is where you type commands. All the following steps use this terminal.

**Step 6 — Install ROS Noetic, AMBF, and ambf_client**

Follow the **Linux** steps in the section above, starting from "Install ROS Noetic". Run every command inside the Ubuntu terminal in VMware.

**Step 7 — Install the GUI package inside the VM**

After completing the Linux steps, install the uncertainty visualiser:

```bash
pip3 install "uncertainty-networks[gui] @ git+https://github.com/X-M-Zhu/uncertainty_propagation_surgical_robotics.git"
```

**Step 8 — Run everything**

Open two terminal windows inside the VM (**Ctrl + Alt + T** twice):

- **Terminal 1** — start AMBF:
  ```bash
  source /opt/ros/noetic/setup.bash
  cd ~/ambf/build
  ./ambf_simulator --launch_file ../ambf/launch.yaml -l 0,1,2
  ```

- **Terminal 2** — launch the GUI:
  ```bash
  uncertainty-gui
  ```

Both windows will appear inside the VMware Fusion window on your Mac. Select a robot and click **Live (AMBF)** to connect to the running simulator.

---

## What this framework can do

| Capability | Method |
|---|---|
| Propagate uncertainty along a single kinematic chain | `query()` |
| Propagate uncertainty and fuse all paths (multi-path) | `query_frame()` |
| Find where a point lands in another frame, with uncertainty | `query_point()` |
| Compute correlation-aware relative vector between two points | `query_relative_vector()` |
| Compute distance between two points with correct correlation | `query_distance()` |
| Apply a loop closure constraint to reduce uncertainty | `query_closed_loop_posterior()` |
| Automatically find and apply all loop constraints | `query_auto_loop_posterior()` |
| Condition on heterogeneous observations (loop, point, distance) | `condition_on_observations()` |

---

## Tutorial

### Concept: what is an uncertain transform?

Every edge in the network stores an `UncertainTransform`: a rigid body transform (4x4 matrix)
together with a 6x6 covariance matrix that describes how uncertain that transform is.

The perturbation model is (CIS I right-multiplicative convention):

```
T_true  =  T_nom  *  Exp(eta)

where  eta = [alpha; epsilon]  in R^6
           alpha = rotation error  (3x1, radians)
           epsilon = translation error  (3x1, metres)

eta ~ N(0, C),   C is the 6x6 covariance matrix
```

---

### Example 1: single kinematic chain (frame to frame)

A robot arm with three links: World → Shoulder → Elbow → Tool.

```python
import numpy as np
from uncertainty_networks import GeometricNetwork, UncertainTransform

def make_edge(translation, sigma):
    """Helper: pure translation edge with isotropic uncertainty."""
    F = np.eye(4)
    F[:3, 3] = translation
    C = sigma**2 * np.eye(6)
    return UncertainTransform(F, C)

net = GeometricNetwork()

net.add_edge("World",    "Shoulder", make_edge([0.0, 0.0, 0.5], sigma=0.002))
net.add_edge("Shoulder", "Elbow",    make_edge([0.3, 0.0, 0.0], sigma=0.003))
net.add_edge("Elbow",    "Tool",     make_edge([0.2, 0.0, 0.0], sigma=0.002))

# Query: what is the transform from World to Tool, and how uncertain is it?
result = net.query("World", "Tool")

print("Path taken:      ", result.path)
print("Nominal position:", result.transform.F_nom[:3, 3])
print("Trace of C:      ", np.trace(result.transform.C).round(6))
```

The covariance accumulates along the chain. Each new edge adds uncertainty via
the adjoint mapping:

```
C_AC  =  Ad_{F_BC^{-1}} * C_AB * Ad_{F_BC^{-1}}^T  +  C_BC
```

---

### Example 2: frame to point

A tool tip is rigidly attached to the Tool frame with a small local uncertainty.
Query its position and uncertainty in the World frame.

```python
import numpy as np
from uncertainty_networks import GeometricNetwork, UncertainTransform

net = GeometricNetwork()
net.add_edge("World", "Tool", make_edge([0.5, 0.0, 0.3], sigma=0.003))

# Attach a point to the Tool frame
p_local = np.array([0.05, 0.0, 0.0])   # 5cm offset along tool axis
Cp_local = (0.001**2) * np.eye(3)       # 1mm local uncertainty

net.add_point("tip", frame="Tool", p_local=p_local, Cp=Cp_local)

# Where is the tip in the World frame, and how uncertain?
p_world, Cp_world = net.query_point("tip", "World")

print("Tip position in World:", p_world.round(4))
print("Tip std dev (mm):     ", np.sqrt(np.diag(Cp_world) * 1e6).round(3), "mm")
```

The point covariance has two contributions:

```
Cp_world  =  J_eta * C_pose * J_eta^T   (from frame pose uncertainty)
           + R * Cp_local * R^T          (from the point's own local uncertainty)
```

where  J_eta = [ -R [p_in]x  |  R ]  is the CIS I right-convention point Jacobian
       and p_in is the point in the source (body) frame.

---

### Example 3: distance between two points (correlation-aware)

Two anatomical landmarks are both expressed in the Camera frame. They share
edges in their kinematic paths, so their uncertainties are correlated.

```python
import numpy as np
from uncertainty_networks import GeometricNetwork, UncertainTransform

net = GeometricNetwork()
net.add_edge("World", "CT",   make_edge([0.1, 0.0, 0.0], sigma=0.002))
net.add_edge("CT",    "BoneA", make_edge([0.0, 0.05, 0.0], sigma=0.001))
net.add_edge("CT",    "BoneB", make_edge([0.0, -0.05, 0.0], sigma=0.001))

net.add_point("landmark_A", "BoneA", np.array([0.0, 0.0, 0.0]), 1e-6*np.eye(3))
net.add_point("landmark_B", "BoneB", np.array([0.0, 0.0, 0.0]), 1e-6*np.eye(3))

# Naive (independent) estimate — overestimates uncertainty
_, C_naive = net.query_relative_vector_independent("landmark_A", "landmark_B", "World")

# Correlation-aware estimate — correct
delta, C_correct = net.query_relative_vector("landmark_A", "landmark_B", "World")

# Distance with variance
d, var_d = net.query_distance("landmark_A", "landmark_B", "World")

print("Distance:             ", round(d * 1000, 2), "mm")
print("Std dev (naive):      ", round(np.sqrt(np.trace(C_naive)) * 1000, 3), "mm")
print("Std dev (correct):    ", round(np.sqrt(np.trace(C_correct)) * 1000, 3), "mm")
```

The shared CT→World edge moves both landmarks together, so their relative
distance is less uncertain than the naive sum suggests.

---

### Example 4: multiple paths — Bayesian fusion

When two different paths connect the same pair of frames, each path is an
independent measurement of the same transform. Bayes' rule fuses them:

```
C_fused  =  inv(  C_1^{-1}  +  C_2^{-1}  +  ...  +  C_N^{-1}  )
```

The fused covariance is strictly smaller than any individual path.

```python
import numpy as np
from uncertainty_networks import GeometricNetwork, UncertainTransform

net = GeometricNetwork()

# Path 1: A -> B -> C
net.add_edge("A", "B", make_edge([1.0, 0.0, 0.0], sigma=0.1))
net.add_edge("B", "C", make_edge([1.0, 0.0, 0.0], sigma=0.1))

# Path 2: A -> D -> C
net.add_edge("A", "D", make_edge([0.0, 1.0, 0.0], sigma=0.1))
net.add_edge("D", "C", make_edge([0.0,-1.0, 2.0], sigma=0.1))

#          A
#         / \
#        B   D
#         \ /
#          C

# query_frame automatically finds both paths and fuses them
result = net.query_frame("A", "C")

print("Number of paths found:", result.n_paths)
for i, pr in enumerate(result.path_results):
    print(f"  Path {i+1}: {pr.path}  trace(C) = {np.trace(pr.transform.C):.4f}")
print("Fused trace(C):      ", np.trace(result.transform.C).round(4))
```

Expected output:
```
Number of paths found: 2
  Path 1: ['A', 'B', 'C']  trace(C) = 0.1400
  Path 2: ['A', 'D', 'C']  trace(C) = 0.1400
Fused trace(C):       0.0650
```

With two equal-uncertainty paths, the fused result has roughly half the uncertainty.

---

### Example 5: loop closure constraint

When a loop exists in the network (e.g. both feet of a human body touch the
ground), the constraint that the loop closes to identity reduces uncertainty
on both paths simultaneously.

```python
import numpy as np
from uncertainty_networks import GeometricNetwork, UncertainTransform

net = GeometricNetwork()
net.add_edge("Pelvis", "L_Hip",  make_edge([-0.1, 0.0, -0.5], sigma=0.005))
net.add_edge("L_Hip",  "L_Foot", make_edge([ 0.0, 0.0, -0.4], sigma=0.005))
net.add_edge("Pelvis", "R_Hip",  make_edge([ 0.1, 0.0, -0.5], sigma=0.005))
net.add_edge("R_Hip",  "R_Foot", make_edge([ 0.0, 0.0, -0.4], sigma=0.005))
net.add_edge("L_Foot", "R_Foot", make_edge([ 0.2, 0.0,  0.0], sigma=0.001))

# Two paths from Pelvis to R_Foot:
path_1 = ["Pelvis", "R_Hip",  "R_Foot"]        # direct
path_2 = ["Pelvis", "L_Hip", "L_Foot", "R_Foot"]  # via left leg

posterior = net.query_closed_loop_posterior(path_1, path_2)

prior = net.query("Pelvis", "R_Foot")
print("Before (path 1 alone): trace =", np.trace(prior.transform.C).round(6))
print("After  (loop conditioning): trace =", np.trace(posterior.C_res).round(6))
```

The loop constraint says the two paths must agree. Conditioning on this
tightens the covariance on both paths.

---

## Running Monte Carlo Validation

All analytic results are validated against Monte Carlo simulation. Run any of:

```bash
python scripts/validate_open_chain_mc.py                  # SE(3) chain propagation
python scripts/validate_frame_to_point_mc.py              # frame-to-point
python scripts/validate_point_mc.py                       # point uncertainty
python scripts/validate_point_to_point_mc.py              # point-to-point correlation
python scripts/validate_closed_loop_mc.py                 # loop constraint conditioning
python scripts/validate_random_network_mc.py              # random network stress test
python scripts/validate_multi_edge_corr_mc_chain.py       # multi-edge correlation (chain)
python scripts/validate_multi_edge_corr_mc_branching.py   # multi-edge correlation (branching)
python scripts/validate_shared_infrastructure_mc.py       # surgical robotics scenario
```

Each script prints analytic vs Monte Carlo covariance and the relative
Frobenius error. Errors below ~1% confirm the first-order approximation is valid.

---

## Project Structure

```
src/uncertainty_networks/
    se3.py                  SE(3) math: exp, log, adjoint, skew
    uncertain_geometry.py   UncertainTransform: compose, inv, transform_point
    network.py              GeometricNetwork: all query methods
    closed_loop.py          Loop conditioning and Gaussian fusion
    observations.py         Observation/factor abstraction (loop, point, distance)
    visualization.py        Static (matplotlib) and interactive (Plotly 3D) visualization
    examples.py             Reusable example network builders

scripts/
    validate_*_mc.py        Monte Carlo validation scripts
    plot_network.py         Generate visualization figures (saved to results/)
    demo_open_chain.py      Simple demo

tests/                      Unit tests (run with pytest)
docs/                       Math note (PDF) and project report
results/                    Generated figures
PSEUDOCODE.md               Full pseudocode and math reference
```

---

## Key Conventions

- All perturbations follow the **CIS I right-multiplicative** convention: `T = T_nom * Exp(eta)`
- Perturbation vector ordering: `eta = [alpha (rotation); epsilon (translation)]`
- Covariance matrices are always 6x6 in this ordering
- Forward propagation and loop conditioning are separate steps by design
- Monte Carlo scripts are for validation only, not production use
