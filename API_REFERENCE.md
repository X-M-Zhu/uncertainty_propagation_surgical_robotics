# API Reference — `uncertainty-networks`

**Author:** X.M. Christine Zhu &nbsp;·&nbsp; **Mentor:** Dr. Russell H. Taylor  
**Mathematical reference:** `docs/math_note.pdf`, `PSEUDOCODE.md`

---

## Installation

```bash
# Core library only
pip install "uncertainty-networks @ git+https://github.com/X-M-Zhu/uncertainty_propagation_surgical_robotics.git"

# Core + GUI + visualisation (recommended for students)
pip install "uncertainty-networks[gui] @ git+https://github.com/X-M-Zhu/uncertainty_propagation_surgical_robotics.git"
```

After installing with `[gui]`, launch the interactive visualiser from any terminal:

```bash
uncertainty-gui
```

---

## Contents

1. [Conventions](#conventions)
2. [UncertainTransform](#uncertaintransform)
3. [GeometricNetwork](#geometricnetwork)
   - [Building the graph](#building-the-graph)
   - [Frame queries](#frame-queries)
   - [Point queries](#point-queries)
   - [Loop-closure queries](#loop-closure-queries)
4. [Observations and joint conditioning](#observations-and-joint-conditioning)
5. [Standalone utilities](#standalone-utilities)
6. [Visualisation](#visualisation)
7. [Return-type reference](#return-type-reference)
8. [Quick-reference table](#quick-reference-table)

---

## Conventions

All transforms use the **CIS I left-multiplicative** perturbation model:

```
T_true  =  Exp(η)  ⊗  T_nom

η = [α; ε]  ∈  R^6
    α  — rotation  error  (3×1, radians)
    ε  — translation error (3×1, metres)

η ~ N(0, C),   C ∈ R^{6×6}
```

- `F_nom` is always a **4×4 homogeneous matrix** (SE(3)).
- `C` is always a **6×6 covariance matrix** with `[rotation | translation]` ordering.
- Covariances accumulate via the SE(3) adjoint:
  `C_ac = C_ab + Ad_{F_ab} C_bc Ad_{F_ab}^T`

---

## UncertainTransform

```python
from uncertainty_networks import UncertainTransform
```

Represents one rigid-body transform with associated uncertainty.

### Constructor

```python
UncertainTransform(F_nom: np.ndarray, C: np.ndarray)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `F_nom` | `(4, 4) ndarray` | Nominal homogeneous transform (must be valid SE(3)) |
| `C` | `(6, 6) ndarray` | Covariance of η = [α; ε]. Must be symmetric positive semi-definite |

```python
import numpy as np
from uncertainty_networks import UncertainTransform

F = np.eye(4)
F[:3, 3] = [0.3, 0.0, 0.0]          # 30 cm translation along x
C = (0.002**2) * np.eye(6)           # 2 mm / 2 mrad isotropic uncertainty
T = UncertainTransform(F, C)
```

### `UncertainTransform.identity(C=None)`

```python
@staticmethod
UncertainTransform.identity(C: np.ndarray | None = None) -> UncertainTransform
```

Returns the identity transform. `C` defaults to zeros if omitted.

---

### `.compose(other)` / `@` operator

```python
T_ac = T_ab.compose(T_bc)
T_ac = T_ab @ T_bc           # shorthand
```

Composes two transforms with first-order covariance propagation.

**Returns:** `UncertainTransform` — composed transform `F_ac` with propagated covariance.

---

### `.inv()`

```python
T_inv = T.inv()
```

Inverts the transform. Covariance is mapped through the adjoint of the inverse:
`C_inv ≈ Ad_{F^{-1}} C Ad_{F^{-1}}^T`

**Returns:** `UncertainTransform`

---

### `.transform_point(p, Cp=None)`

```python
p_out, Cp_out = T.transform_point(p, Cp=None)
```

Transforms a 3D point and propagates its uncertainty.

| Parameter | Type | Description |
|-----------|------|-------------|
| `p` | `(3,) array-like` | Input point in source frame |
| `Cp` | `(3, 3) ndarray` or `None` | Intrinsic point covariance (optional) |

**Returns:**
- `p_out` — `(3,)` nominal transformed point
- `Cp_out` — `(3, 3)` propagated point covariance (pose-induced + intrinsic if provided)

---

## GeometricNetwork

```python
from uncertainty_networks import GeometricNetwork
```

A directed graph of coordinate frames connected by `UncertainTransform` edges. The main class you will use.

### Constructor

```python
net = GeometricNetwork()
```

---

## Building the graph

### `add_frame(name)`

```python
net.add_frame(name: str) -> None
```

Adds an isolated frame node. Frames are also created automatically by `add_edge()`.

---

### `add_edge(src, dst, T_src_dst, ...)`

```python
edge_id = net.add_edge(
    src:          str,
    dst:          str,
    T_src_dst:    UncertainTransform,
    add_inverse:  bool = True,
    is_certain:   bool = False,
    edge_type:    str  = "se3",
) -> str
```

Adds a directed edge `src → dst`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `str` | Source frame name (created if new) |
| `dst` | `str` | Destination frame name (created if new) |
| `T_src_dst` | `UncertainTransform` | Transform from src to dst |
| `add_inverse` | `bool` | Also add the reverse edge (same `edge_id`). Default `True` |
| `is_certain` | `bool` | Mark as perfectly known (C ≈ 0). Default `False` |
| `edge_type` | `str` | `"se3"` (full), `"rot_only"`, `"trans_only"`, or `"vector"`. Default `"se3"` |

**Returns:** `edge_id` — unique string identifier for this physical edge. Shared by forward and inverse directions; used internally to track correlated paths.

```python
import numpy as np
from uncertainty_networks import GeometricNetwork, UncertainTransform

net = GeometricNetwork()

F = np.eye(4); F[:3, 3] = [0.5, 0.0, 0.0]
T = UncertainTransform(F, (0.003**2) * np.eye(6))

net.add_edge("World", "Tool", T)
```

---

### `add_point(name, frame, p_local, Cp)`

```python
net.add_point(
    name:    str,
    frame:   str,
    p_local: np.ndarray,   # (3,)
    Cp:      np.ndarray,   # (3, 3)
) -> None
```

Registers a named 3D point rigidly attached to `frame`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique point identifier |
| `frame` | `str` | Frame the point lives in (must already exist) |
| `p_local` | `(3,) ndarray` | Point coordinates in `frame` |
| `Cp` | `(3, 3) ndarray` | Intrinsic covariance of the point in `frame` |

```python
net.add_point("tip", frame="Tool",
              p_local=np.array([0.05, 0.0, 0.0]),
              Cp=(0.001**2) * np.eye(3))
```

---

## Frame queries

### `query(start, goal)` — single shortest path

```python
result = net.query(start: str, goal: str) -> PathResult
```

Propagates uncertainty along the **BFS shortest path** only. Use this for simple chains where only one path exists.

**Returns:** [`PathResult`](#pathresult)

```python
result = net.query("World", "Tool")
print(result.transform.F_nom[:3, 3])    # nominal position
print(np.trace(result.transform.C))     # total uncertainty (trace)
```

---

### `query_frame(start, goal)` — fused multi-path query *(recommended)*

```python
result = net.query_frame(
    start:     str,
    goal:      str,
    max_depth: int | None = None,
) -> FusedQueryResult
```

Finds **all simple paths** from `start` to `goal` and fuses them using the unified Gaussian linear system (information form). This is the correct method for any network topology.

- One path → same as `query()`.
- Multiple independent paths → Bayesian information fusion: `C_fused = (Σ C_k^{-1})^{-1}`.
- Multiple paths with **shared edges** → off-diagonal blocks in the stacked covariance matrix prevent double-counting. Result is always correct.

| Parameter | Type | Description |
|-----------|------|-------------|
| `start` | `str` | Source frame |
| `goal` | `str` | Goal frame |
| `max_depth` | `int` or `None` | Limit path search depth for large dense networks |

**Returns:** [`FusedQueryResult`](#fusedqueryresult)

```python
result = net.query_frame("World", "Tool")
print("Paths used:", result.n_paths)
print("Fused C trace:", np.trace(result.transform.C))
```

---

### `query_all_paths(start, goal)` — per-path diagnostics

```python
path_results = net.query_all_paths(
    start:     str,
    goal:      str,
    max_depth: int | None = None,
) -> list[PathResult]
```

Returns one `PathResult` per simple path, without fusion. Useful for comparing individual path uncertainties.

---

## Point queries

### `query_point(point_name, target_frame)`

```python
p_nom, Cp = net.query_point(
    point_name:   str,
    target_frame: str,
) -> tuple[np.ndarray, np.ndarray]
```

Transforms a registered point into `target_frame` and propagates its full uncertainty (pose + intrinsic).

**Returns:**
- `p_nom` — `(3,)` nominal position in `target_frame`
- `Cp` — `(3, 3)` covariance in `target_frame`

```python
p_world, Cp_world = net.query_point("tip", "World")
std_mm = np.sqrt(np.diag(Cp_world)) * 1000
print("Tip std dev (mm):", std_mm.round(3))
```

---

### `query_relative_vector(src_point, dst_point, query_frame)` *(correlation-aware)*

```python
delta, C_delta = net.query_relative_vector(
    src_point:   str,
    dst_point:   str,
    query_frame: str,
) -> tuple[np.ndarray, np.ndarray]
```

Relative vector `dst − src` expressed in `query_frame`, with **correct cross-covariance** from shared edges.

Two points that share upstream edges (e.g. both attached to the same robot's kinematic chain) are correlated — their relative position is less uncertain than the naive sum of individual uncertainties. This method accounts for that.

**Returns:**
- `delta` — `(3,)` nominal relative vector
- `C_delta` — `(3, 3)` covariance of the relative vector

---

### `query_relative_vector_independent(src_point, dst_point, query_frame)` *(upper bound)*

```python
delta, C_delta = net.query_relative_vector_independent(
    src_point:   str,
    dst_point:   str,
    query_frame: str,
) -> tuple[np.ndarray, np.ndarray]
```

Same as above but treats the two points as independent (`C_delta = C_src + C_dst`). This **overestimates** uncertainty when edges are shared. Useful as a conservative upper bound or sanity check.

---

### `query_distance(src_point, dst_point, query_frame)`

```python
d, var_d = net.query_distance(
    src_point:   str,
    dst_point:   str,
    query_frame: str,
) -> tuple[float, float]
```

Euclidean distance between two points with first-order variance, using the correlation-aware relative vector.

**Returns:**
- `d` — nominal distance (metres)
- `var_d` — first-order variance of the distance (metres²)

```python
d, var_d = net.query_distance("landmark_A", "landmark_B", "World")
print(f"Distance: {d*1000:.2f} mm  ±  {np.sqrt(var_d)*1000:.3f} mm")
```

---

## Loop-closure queries

When two different paths connect the same pair of frames (a loop), constraining them to agree reduces uncertainty on both.

### `query_closed_loop_posterior(path_res, path_k)`

```python
posterior = net.query_closed_loop_posterior(
    path_res: list[str],   # reference path  (frame names)
    path_k:   list[str],   # alternative path (frame names)
) -> LoopPosterior
```

Applies **one loop-closure constraint**: conditions on the requirement that `path_res` and `path_k` produce the same transform.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path_res` | `list[str]` | Reference path as a list of frame names |
| `path_k` | `list[str]` | Alternative path sharing the same start and end frames |

**Returns:** [`LoopPosterior`](#loopposterior)

```python
posterior = net.query_closed_loop_posterior(
    ["Pelvis", "R_Hip", "R_Foot"],
    ["Pelvis", "L_Hip", "L_Foot", "R_Foot"],
)
print("After loop conditioning, C_res trace:", np.trace(posterior.C_res))
```

---

### `query_auto_loop_posterior(start, goal)`

```python
posterior = net.query_auto_loop_posterior(
    start: str,
    goal:  str,
) -> LoopPosterior
```

Automatically discovers all independent loop constraints between `start` and `goal` (via spanning-tree cycle basis) and applies them simultaneously.

**Returns:** [`LoopPosterior`](#loopposterior)

---

## Observations and joint conditioning

For heterogeneous measurements (loop closures, observed 3D points, measured distances) applied jointly in one Bayesian update step.

```python
from uncertainty_networks import (
    LoopObservation,
    PointObservation,
    DistanceObservation,
    condition_on_observations,
)
```

---

### `LoopObservation`

```python
LoopObservation(
    net:      GeometricNetwork,
    path_res: list[str],
    path_k:   list[str],
    C_nu:     np.ndarray,    # (6, 6) loop noise covariance
)
```

A 6-DOF loop-closure constraint: the two paths must produce the same SE(3) transform.

---

### `PointObservation`

```python
PointObservation(
    net:          GeometricNetwork,
    point_name:   str,
    observed_pos: np.ndarray,   # (3,) observed position in measurement_frame
    measurement_frame: str,
    C_nu:         np.ndarray,   # (3, 3) observation noise covariance
)
```

A 3-DOF constraint: an external sensor observed a registered point at a known position.

---

### `DistanceObservation`

```python
DistanceObservation(
    net:       GeometricNetwork,
    src_point: str,
    dst_point: str,
    query_frame: str,
    observed_distance: float,
    var_nu:    float,           # scalar observation noise variance
)
```

A 1-DOF constraint: an external sensor measured the distance between two registered points.

---

### `condition_on_observations(state_covs, observations)`

```python
result = condition_on_observations(
    state_covs:   dict[str, np.ndarray],   # frame_name -> (6, 6) prior covariance
    observations: list[Observation],
) -> ConditioningResult
```

Applies all observations jointly in one information-filter update:
`C_post = (C0^{-1} + H^T C_ν^{-1} H)^{-1}`

| Parameter | Type | Description |
|-----------|------|-------------|
| `state_covs` | `dict[str, (6,6) ndarray]` | Prior covariance for each state frame |
| `observations` | `list[Observation]` | Any mix of `LoopObservation`, `PointObservation`, `DistanceObservation` |

**Returns:** [`ConditioningResult`](#conditioningresult)

---

## Standalone utilities

```python
from uncertainty_networks import condition_on_loop, fuse_gaussian_covs
```

### `condition_on_loop(T_res, T_k, C_nu)`

```python
posterior = condition_on_loop(
    T_res: UncertainTransform,
    T_k:   UncertainTransform,
    C_nu:  np.ndarray,          # (6, 6) loop noise covariance
) -> LoopPosterior
```

Low-level single loop conditioning without requiring a `GeometricNetwork`. Takes two `UncertainTransform` objects directly.

---

### `fuse_gaussian_covs(covs)`

```python
C_fused = fuse_gaussian_covs(covs: list[np.ndarray]) -> np.ndarray
```

Information-form fusion of N independent Gaussian covariances:
`C_fused = (Σ C_k^{-1})^{-1}`

```python
C_fused = fuse_gaussian_covs([C1, C2, C3])
```

---

## Visualisation

```python
from uncertainty_networks.visualization import (
    plot_network_static,
    plot_network_interactive,
)
```

Requires the `[gui]` install: `pip install uncertainty-networks[gui]`

---

### `plot_network_static(net, reference_frame, ...)`

```python
fig = plot_network_static(
    net:             GeometricNetwork,
    reference_frame: str,
    ellipsoid_sigma: float = 3.0,
    figsize:         tuple = (10, 8),
) -> matplotlib.figure.Figure
```

Static matplotlib figure. Frames shown as labelled points; edges coloured by uncertainty magnitude; uncertainty ellipsoids drawn at `ellipsoid_sigma` standard deviations.

---

### `plot_network_interactive(net, reference_frame, ...)`

```python
fig = plot_network_interactive(
    net:             GeometricNetwork,
    reference_frame: str,
    ellipsoid_sigma: float = 3.0,
    title:           str   = "Geometric Network",
) -> plotly.graph_objects.Figure
```

Interactive Plotly 3D figure with rotating camera, hover cards showing frame names and uncertainty trace, and colour-coded uncertainty ellipsoids.

```python
fig = plot_network_interactive(net, "World", ellipsoid_sigma=20)
fig.show()
```

---

## Return-type reference

### `PathResult`

| Field | Type | Description |
|-------|------|-------------|
| `path` | `list[str]` | Frame names along the path, e.g. `["World", "Shoulder", "Tool"]` |
| `transform` | `UncertainTransform` | Composed nominal transform and covariance along this path |
| `edge_ids` | `list[str]` | Edge ID for each hop (length = `len(path) − 1`) |
| `certain_mask` | `list[bool]` | `True` if edge is marked certain (C ≈ 0) |
| `forward_mask` | `list[bool]` | `True` if edge was traversed in its canonical direction |
| `edge_types` | `list[str]` | `"se3"`, `"rot_only"`, `"trans_only"`, or `"vector"` per hop |

---

### `FusedQueryResult`

| Field | Type | Description |
|-------|------|-------------|
| `transform` | `UncertainTransform` | Best-estimate transform with fused covariance C₀ |
| `n_paths` | `int` | Number of simple paths found and fused |
| `path_results` | `list[PathResult]` | Per-path results before fusion (for diagnostics) |

---

### `LoopPosterior`

| Field | Type | Description |
|-------|------|-------------|
| `C_res` | `(6, 6) ndarray` | Posterior covariance of the reference path transform |
| `C_k` | `(6, 6) ndarray` | Posterior covariance of the alternative path transform |
| `C_cross` | `(6, 6) ndarray` | Cross-covariance between the two paths |
| `C_full` | `(12, 12) ndarray` | Full joint posterior covariance `[C_res, C_cross; C_cross^T, C_k]` |

---

### `ConditioningResult`

| Field | Type | Description |
|-------|------|-------------|
| `posterior_covs` | `dict[str, (6,6) ndarray]` | Posterior covariance for each state frame |
| `cross_cov(a, b)` | `(6,6) ndarray` | Cross-covariance between frames `a` and `b` (method call) |

---

## Quick-reference table

| Goal | Method | Returns |
|------|--------|---------|
| Uncertainty along a single kinematic chain | `net.query(start, goal)` | `PathResult` |
| Uncertainty fused over all paths *(recommended)* | `net.query_frame(start, goal)` | `FusedQueryResult` |
| Where is a named point, with uncertainty? | `net.query_point(name, frame)` | `(p, Cp)` |
| Relative vector between two points (correct) | `net.query_relative_vector(a, b, frame)` | `(delta, C_delta)` |
| Relative vector, independence approximation | `net.query_relative_vector_independent(a, b, frame)` | `(delta, C_delta)` |
| Distance between two points with variance | `net.query_distance(a, b, frame)` | `(d, var_d)` |
| Apply one loop-closure constraint | `net.query_closed_loop_posterior(path1, path2)` | `LoopPosterior` |
| Apply all loop constraints automatically | `net.query_auto_loop_posterior(start, goal)` | `LoopPosterior` |
| Joint conditioning on mixed observations | `condition_on_observations(covs, obs)` | `ConditioningResult` |
| Fuse N independent Gaussian covariances | `fuse_gaussian_covs([C1, C2, ...])` | `(6,6) ndarray` |
| Static network visualisation | `plot_network_static(net, frame)` | `matplotlib Figure` |
| Interactive 3D network visualisation | `plot_network_interactive(net, frame)` | `plotly Figure` |
