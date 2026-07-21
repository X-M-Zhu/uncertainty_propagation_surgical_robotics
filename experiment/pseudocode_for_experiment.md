# Pseudocode for All Experiments
**Author:** X.M. Christine Zhu

---

## Part 0 — Hardware Data Collection

**Source:** `utils/atracsys_interface.py`, `exp1_fixed_drill/collect.py`,
`exp2_both_move/collect.py`

The Atracsys FusionTrack tracker publishes two ROS topics per rigid body:

```
/atracsys/<body>/measured_cp        →  fitted rigid-body pose  T ∈ SE(3)   (4×4)
/atracsys/<body>/marker_positions   →  raw per-fiducial 3-D positions       (n×3)
```

`measured_cp` is the pose of the body's coordinate frame in the tracker
frame.  `marker_positions` gives the raw 3-D position of every detected
fiducial (reflective ball) independently of the pose fit.

**Collection loop (N = 300 samples per position):**

```
FOR i = 1 to N:
    T_A_i  ← measured_cp("Anspoch_drill")   shape (4,4)   SE(3), metres
    T_B_i  ← measured_cp("Anatomy")          shape (4,4)   SE(3), metres
    mk_A_i ← marker_positions("Anspoch_drill")  shape (n_markers, 3)   metres
    mk_B_i ← marker_positions("Anatomy")        shape (n_markers, 3)   metres

SAVE:
    bodyA.csv           ← (N, 4, 4) flattened    fitted poses of body A
    bodyB.csv           ← (N, 4, 4) flattened    fitted poses of body B
    markersA_raw.csv    ← (N, n_markers × 3)     raw fiducial positions of body A
    markersB_raw.csv    ← (N, n_markers × 3)     raw fiducial positions of body B
```

*All timestamps are simultaneous within each frame i — this matters for the
center-of-frame analysis (camera drift cancels).*

---

## Part 1 — Per-Marker Noise  C_ball

**Source:** `exp1_fixed_drill/analyze_marker_covariances.py`,
`exp2_both_move/analyze_marker_covariances.py`

**Produces:** `fig_sigma_ball_vs_distance.png`, `fig_sigma_ball_by_config.png`,
`fig_marker_isotropy.png`, `shared_cal/C_ball.json`

### 1a — Load raw marker data

```
markers_A ← load_marker_csv("markersA_raw.csv")    shape (N=300, n_markers=4, 3)   metres
markers_B ← load_marker_csv("markersB_raw.csv")    shape (N=300, n_markers=4, 3)   metres
```

### 1b — Per-marker 3×3 covariance  (function `marker_covariance_stats`)

For each body and for each fiducial index k = 0, 1, 2, 3:

```
pts_k  = markers[:, k, :]                      shape (N, 3)   metres

── Mean position ────────────────────────────────────────────────────────────
μ_k  = (1/N) Σᵢ pts_k[i]                       shape (3,)     metres

── 3×3 empirical covariance ─────────────────────────────────────────────────
C_k  = (1/(N-1))  Σᵢ (pts_k[i] − μ_k)(pts_k[i] − μ_k)ᵀ    shape (3,3)   metres²

     implemented as:   C_k = np.cov(pts_k, rowvar=False)

── Scalar noise σ (one number summarising C_k) ──────────────────────────────
σ_k  = sqrt( trace(C_k) / 3 ) × 1000           mm

     This is the RMS position error averaged equally over x, y, z.

── Eigendecomposition of C_k ────────────────────────────────────────────────
[λ₁ ≤ λ₂ ≤ λ₃], [v₁, v₂, v₃] = eigh(C_k)     eigenvalues in metres², ascending

── Isotropy ──────────────────────────────────────────────────────────────────
isotropy_k = λ₁ / λ₃

    → 1.0   noise ellipsoid is a perfect sphere  (equal in all 3 directions)
    → ~0    noise is needle-shaped  (dominated by one axis)
```

### 1c — Average C_ball  (representative noise for OpticalTracker model)

```
C_ball = (1 / (n_positions × n_markers))  Σ_{all k, all positions}  C_k

σ_ball = sqrt( trace(C_ball) / 3 ) × 1000     mm

Saved to  shared_cal/C_ball.json  for use by the OpticalTracker simulator.
```

### 1d — What the figures show

**`fig_sigma_ball_vs_distance.png` (exp1)**
- X-axis: distance from tracker camera to drill body centroid (mm)
  computed as `|μ_drill_centroid|` in metres × 1000
- Y-axis: per-marker σ_k (mm)
- One line per fiducial (4 lines per body, 2 bodies)
- Horizontal dashed line: σ_ball (average across all)

**`fig_marker_isotropy.png` (both exp1 and exp2)**
- X-axis: position / configuration label
- Y-axis: isotropy  λ₁/λ₃  (0 to 1)
- Bars: average isotropy across all 4 markers for each body
- Dots: individual fiducial values overlaid on bars
- Dashed horizontal line at 1.0 = perfect isotropy

---

## Part 2 — Center-of-Frame Analysis

**Source:** `exp1_fixed_drill/analyze.py`, `utils/se3_stats.py`,
`src/uncertainty_networks/uncertain_geometry.py`, `src/uncertainty_networks/se3.py`

**Produces:** `fig_sigma_vs_drill_distance.png`

The figure shows three σ curves vs. drill distance from the tracker:
- **Gray** — σ_TA   (drill body noise as seen from tracker)
- **Red**  — σ_AB empirical  (directly observed relative pose noise)
- **Blue** — σ_AB predicted  (computed from C_TA, C_TB by the framework)

### 2a — SE(3) mathematics used throughout

**SE(3) element:**  T = [ R  p ; 0  1 ]   where R ∈ SO(3), p ∈ R³

**Inverse:**
```
inv(T) = [ Rᵀ   −Rᵀp ]
         [  0      1  ]
```

**Adjoint:**
```
Ad(T) = [    R      |   0  ]    shape (6, 6)
        [ [p]× R    |   R  ]

where [p]× is the 3×3 skew-symmetric matrix of p:

    [p]× = [  0   −p₃   p₂ ]
           [ p₃    0   −p₁ ]
           [−p₂   p₁    0  ]

The bottom-left block is  [p]× R  (skew matrix times R, NOT R times skew).
```

**Log map  log_SE3(T):**
```
α   = log_SO3(R)          ← rotation vector, shape (3,)
ε   = J⁻¹(α) p            ← NOT simply p; goes through inverse left Jacobian of SO(3)
ξ   = [ α ; ε ]           ← twist, shape (6,)   convention: rotation first, then translation

For small rotations  J⁻¹ ≈ I  so  ε ≈ p.
```

**Exp map  exp_SE3(ξ):**  inverse of log_SE3, not detailed here.

### 2b — Fréchet mean on SE(3)

*Function `se3_mean` in `utils/se3_stats.py`*

```
INPUT:  samples = {T₁, T₂, ..., T_N}   each Tᵢ ∈ SE(3)

μ ← T₁                         initialise

REPEAT until ||δ|| < tolerance:
    FOR i = 1 to N:
        ξᵢ = log_SE3( μ⁻¹ @ Tᵢ )     ← tangent vector at μ pointing to Tᵢ

    δ = (1/N) Σᵢ ξᵢ                   ← mean tangent vector

    μ ← μ @ exp_SE3( δ )              ← geodesic step on SE(3)

OUTPUT: μ   ← Fréchet mean
```

### 2c — 6×6 empirical covariance on SE(3)

*Function `se3_empirical_stats` in `utils/se3_stats.py`*

```
INPUT:  samples = {T₁, ..., T_N}

Step 1 — compute Fréchet mean μ  (see 2b above)

Step 2 — compute tangent-space residuals
    FOR i = 1 to N:
        ηᵢ = log_SE3( μ⁻¹ @ Tᵢ )     shape (6,)
           = [ αᵢ  ]   rotation part     (3,)
             [ εᵢ  ]   translation part  (3,)

Step 3 — 6×6 sample covariance
    C = (1/(N−1))  Σᵢ  ηᵢ ηᵢᵀ         shape (6,6)

        [  C_rot    C_cross  ]   rows/cols 0:3 = rotation
        [  C_cross  C_trans  ]   rows/cols 3:6 = translation

    Symmetrized: C ← 0.5 (C + Cᵀ)

OUTPUT: μ, C
```

This is applied twice:
- `se3_empirical_stats(samples_A)` → **μ_A, C_TA**  (drill body)
- `se3_empirical_stats(samples_B)` → **μ_B, C_TB**  (anatomy body)

### 2d — Scalar σ extracted from C

```
σ_TA (mm) = sqrt( trace( C_TA[3:6, 3:6] ) / 3 ) × 1000

                          ↑
              translation block only (rows 3-5, cols 3-5)
              units of C_TA[3:,3:] are metres²
```

This is plotted as the **gray** line.

### 2e — Empirical C_AB from simultaneous pairs (RED line)

```
FOR i = 1 to N:
    T_AB_i  =  inv(T_TA_i) @ T_TB_i        ← relative pose at sample i

μ_AB, C_AB_emp = se3_empirical_stats( {T_AB_1, ..., T_AB_N} )

σ_AB_emp (mm) = sqrt( trace( C_AB_emp[3:6, 3:6] ) / 3 ) × 1000
```

Because T_TA_i and T_TB_i are measured at the **same instant** i, the
camera's slow drift appears in both equally and cancels in the ratio
`inv(T_TA_i) @ T_TB_i`.

### 2f — Predicted C_AB from the uncertainty framework (BLUE line)

*Source: `inv()` and `compose()` in `src/uncertainty_networks/uncertain_geometry.py`*

The RIGHT perturbation convention is used throughout:
```
T_true = F_nom  @  Exp(η),      η ~ N(0, C)
```
Under this convention, propagation rules are:

**Rule 1 — Inverse:**
```
If  T ~ (F_nom, C)  [RIGHT]

Then  inv(T) ~ (F_nom⁻¹, C_inv)  [RIGHT]

where:   C_inv = Ad(F_nom) @ C @ Ad(F_nom)ᵀ

         Note: uses adjoint of F_nom itself, NOT of its inverse.
```

**Rule 2 — Composition (both RIGHT):**
```
If  T_AB ~ (F_AB, C_AB)  [RIGHT]
    T_BC ~ (F_BC, C_BC)  [RIGHT]

Then  T_AC = T_AB @ T_BC ~ (F_AC, C_AC)  [RIGHT]

where:   F_AC  = F_AB @ F_BC
         C_AC  = Ad(F_BC⁻¹) @ C_AB @ Ad(F_BC⁻¹)ᵀ  +  C_BC

         Note: uses adjoint of F_BC inverse.
```

**Applying these two rules to predict C_AB:**

```
Step 1 — invert T_TA:

    F_TA_inv = inv_SE3(F_TA)              ← nominal inverse

    C_TA_inv = Ad(F_TA) @ C_TA @ Ad(F_TA)ᵀ    ← propagated covariance

Step 2 — compose  inv(T_TA) @ T_TB:

    F_AB = F_TA_inv @ F_TB                ← nominal composition

    C_AB_pred = Ad(F_TB⁻¹) @ C_TA_inv @ Ad(F_TB⁻¹)ᵀ  +  C_TB

Substituting Step 1 into Step 2 gives the full formula:

    C_AB_pred = Ad(F_TB⁻¹) @ Ad(F_TA) @ C_TA @ Ad(F_TA)ᵀ @ Ad(F_TB⁻¹)ᵀ  +  C_TB

σ_AB_pred (mm) = sqrt( trace( C_AB_pred[3:6, 3:6] ) / 3 ) × 1000
```

### 2g — Why predicted σ_AB >> empirical σ_AB

**The gap is not a bug.** It is a known limitation of the measurement setup.

```
C_TA  =  C_random_noise  +  C_camera_drift
            (small)               (large, slow)

C_camera_drift:  the tracker camera itself moves slightly over the
                 300-sample collection window.  All bodies shift together
                 with the camera — this shift is absorbed into C_TA.

C_AB_emp:  computed from inv(T_TA_i) @ T_TB_i at the same instant.
           Camera drift appears in T_TA_i and T_TB_i with the same sign
           and cancels in the ratio.  C_AB_emp ≈ C_random_noise only.

C_AB_pred:  computed from C_TA which contains C_camera_drift.
            The adjoint formula then amplifies the rotation part of the
            drift by the lever arm  |p_TA| ≈ 1.5 m (distance from
            camera to drill body), inflating the predicted translation
            noise by 10–20×.
```

The framework formula is mathematically correct.  The problem is that
C_TA is the wrong input to the formula for this setup.  The correct fix
is to first factor out camera motion (`analyze_ei.py`) and feed only the
residual per-body noise into the framework.

---

## Summary: Formula Behind Each Figure

| Figure | Formula | Script |
|--------|---------|--------|
| σ_ball vs distance | `σ = sqrt(trace(C_k)/3)*1000`, `C_k = cov(pts_k)` | `analyze_marker_covariances.py` |
| Isotropy | `λ_min / λ_max` from `eigh(C_k)` | `analyze_marker_covariances.py` |
| σ_TA (gray) | `sqrt(trace(C_TA[3:,3:])/3)*1000`, C_TA from se3_empirical_stats | `analyze.py` |
| σ_AB emp (red) | same formula on C from `se3_empirical_stats({inv(T_A_i)@T_B_i})` | `analyze.py` |
| σ_AB pred (blue) | `sqrt(trace(C_AB_pred[3:,3:])/3)*1000`, C_AB_pred from two-step adjoint propagation | `analyze.py` |
