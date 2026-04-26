# Simulation-Based Uncertainty Propagation in Geometric Networks for Surgical Robotics

**Author:** X.M. Christine Zhu
**Mentor:** Dr. Russell H. Taylor

This repository implements and validates a mathematical framework for **uncertainty propagation in geometric networks**, following the CIS I left-multiplicative perturbation convention.

Built for surgical robotics applications where multiple sensors, rigid links, and coordinate
frames form a network, and you need to know how measurement errors travel through that network
to affect a final quantity (e.g. tool-tip position, distance between two anatomical landmarks).

The math is documented in `docs/` and `PSEUDOCODE.md`.

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

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/kinematic-uncertainty-networks.git
cd kinematic-uncertainty-networks
```

### 2. Create and activate a Python environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# or
venv\Scripts\activate           # Windows
```

### 3. Install the package

```bash
pip install -e .
```

### 4. Verify

```bash
pytest
```

All tests should pass. They cover SE(3) operations, covariance propagation, point uncertainty,
network queries, and closed-loop conditioning.

---

## Tutorial

### Concept: what is an uncertain transform?

Every edge in the network stores an `UncertainTransform`: a rigid body transform (4x4 matrix)
together with a 6x6 covariance matrix that describes how uncertain that transform is.

The perturbation model is (CIS I left-multiplicative convention):

```
T_true  =  Exp(eta)  *  T_nom

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
C_AC  =  C_AB  +  Ad_{F_AB} * C_BC * Ad_{F_AB}^T
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

where  J_eta = [ -[p_nom]x  |  I_3 ]  is the CIS I point Jacobian.

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

- All perturbations follow the **CIS I left-multiplicative** convention: `T = Exp(eta) * T_nom`
- Perturbation vector ordering: `eta = [alpha (rotation); epsilon (translation)]`
- Covariance matrices are always 6x6 in this ordering
- Forward propagation and loop conditioning are separate steps by design
- Monte Carlo scripts are for validation only, not production use