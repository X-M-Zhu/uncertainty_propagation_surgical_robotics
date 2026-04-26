# TODO

This repository is an implementation and verification of the mathematical framework defined in:

**“Mathematical Framework for Uncertainty Propagation in Geometric Networks”**

Scope includes:
- forward uncertainty propagation (SE(3) + points + networks),
- closed-loop conditioning (constraint),
- Monte Carlo validation.

---

## Phase 0 — Repo baseline

- [x] Create repository
- [x] Create folders: `src/`, `tests/`, `scripts/`, `docs/`
- [x] Add math note PDF to `docs/`
- [x] Add `.gitignore`
- [x] Add `pyproject.toml`
- [x] Verify package installs locally
- [x] Add README describing scope/goals

---

## Phase 1 — SE(3) utilities (CIS I convention)

📁 `src/uncertainty_networks/se3.py`

- [x] `skew(w)`
- [x] `inv_se3(T)`
- [x] `adjoint_se3(T)`
- [x] `is_se3(T)` sanity checks
- [x] `exp_se3(xi)`
- [x] `log_se3(T)`
- [x] Tests: exp/log roundtrip, identity behavior, shape checks

---

## Phase 2 — UncertainTransform primitive

📁 `src/uncertainty_networks/uncertain_geometry.py`

- [x] `UncertainTransform(F_nom, C)`
- [x] Input validation + covariance symmetry
- [x] `compose()` via adjoint propagation
- [x] `inv()` via first-order inversion
- [x] Operator `@` for composition
- [x] Tests: composition/inversion sanity

---

## Phase 3 — Point propagation (CIS I Jacobians)

📁 `src/uncertainty_networks/uncertain_geometry.py`

- [x] Point transform with pose uncertainty
- [x] Point transform with pose + intrinsic point covariance
- [x] CIS I Jacobian  
  \( J_\eta = [-[p'_{\text{nom}}]_\times\;\;I] \)
- [x] Tests: rotation-only vs translation-only behavior
- [x] Monte Carlo validation script

📁 `scripts/validate_point_mc.py`

---

## Phase 4 — GeometricNetwork (frames + points + path queries)

📁 `src/uncertainty_networks/network.py`

- [x] Directed graph of frames
- [x] Edges store uncertain transforms (+ inverse edges)
- [x] BFS path finding
- [x] Query transform along path
- [x] Add points (attached to frames)
- [x] Query point covariance across frames
- [x] Query point-to-point delta (independent + corr-aware)
- [x] Tests for path queries + point queries

---

## Phase 5 — Monte Carlo validation (forward propagation)

📁 `scripts/`

- [x] Open-chain SE(3) covariance MC validation
- [x] Point MC validation
- [x] Random-network MC harness (many random queries)

---

## Phase 6 — Closed-loop conditioning module

📁 `src/uncertainty_networks/closed_loop.py`

- [x] Loop residual  
  \( r = \log(T_{\text{res}}^{-1}T_k) \)
- [x] FD Jacobians w.r.t. \(\eta_{\text{res}}, \eta_k\)
- [x] Gaussian conditioning (information form)
- [x] Subspace conditioning:
  - [x] rotation-only (\(\alpha\))
  - [x] translation-only (\(\epsilon\))
- [x] Unit tests: posterior covariance reduction
- [x] Monte Carlo validation with importance weighting
- [x] ESS stabilization via auto-tuned residual noise

📁 `scripts/validate_closed_loop_mc.py`

---

## Phase 7 — Closed-loop integrated with the network

📁 `src/uncertainty_networks/network.py`

- [x] Compose transform along explicit paths
- [x] Closed-loop posterior from two alternative paths
- [x] Network-level API for closed-loop queries
- [x] Unit tests for network closed-loop inference

---

## Phase 8 — Multiple constraints (batch closed-loop / CIS HW3)

**Goal:** support conditioning on multiple loop constraints simultaneously.

📁 `src/uncertainty_networks/closed_loop.py`

- [x] Implement `condition_on_multiple_loops(...)` for N simultaneous constraints
- [x] Stack residual Jacobians \(H_i\) and noises \(C_{\nu,i}\)
- [x] Perform joint information update
- [x] Return posterior covariance blocks (`MultiLoopPosterior`)
- [x] Support per-constraint subspace selection (via `condition_on_loop_subspace`)

📁 `tests/test_batch_eval.py`

- [x] Posterior trace decreases as constraints are added
- [x] Covariance symmetry + PSD checks

---

## Phase 9 — Automatic loop discovery in the network

📁 `src/uncertainty_networks/network.py`

- [x] Find all simple paths between the same endpoints (`find_all_paths`)
- [x] Graceful failure if no alternative path exists
- [x] Spanning-tree cycle basis for independent loop discovery (`find_independent_loops`)
- [x] Start/end aware loop filtering and orientation
- [x] Automatic multi-loop posterior (`query_auto_loop_posterior`)

---

## Phase 10 — Observation / factor abstraction

📁 `src/uncertainty_networks/observations.py`

- [x] `Observation` abstract base class (residual, jacobians, noise_cov)
- [x] `LoopObservation` implementation
- [x] `PointObservation` implementation (with CIS I Jacobian auto-build)
- [x] `DistanceObservation` implementation (scalar distance factor)
- [x] `condition_on_observations` — unified joint information filter
- [x] `ConditioningResult` with `cross_cov()` helper

📁 `tests/test_observations.py`

- [x] Residual/Jacobian correctness for all 3 factor types
- [x] `condition_on_observations` matches `condition_on_loop` exactly (single loop)
- [x] Trace decreases as observations are stacked
- [x] Posterior symmetry and PD checks
- [x] Mixed observation types (loop + point + distance)
- [x] `KeyError` on unknown state key
- [x] `cross_cov()` correctness

---

## Phase 11 — Usability and project polish

📁 `README.md`

- [x] How to run tests and validations
- [x] Explain closed-loop capability

📁 `scripts/run_all_validations.py`

- [x] Run all key validation scripts
- [x] Print summary table (errors, traces, ESS)

---

## Phase 12 — Visualization

📁 `scripts/plot_network.py`

- [x] Fig 1: Plot full network graph with edge uncertainty colormap (√tr(C) per edge)
- [x] Fig 2: Highlight a query path + bar chart of cumulative uncertainty budget per hop
- [x] Fig 3: Loop closure — covariance ellipses (tx–ty) before and after conditioning,
             with percentage uncertainty reduction annotated
- [x] Figures saved to `results/`

---

## Out of scope (for now)

- [ ] Real sensor data / ROS runtime
- [ ] AMBF integration
- [ ] Full state estimation / SLAM
- [ ] Advanced cross-covariance models

