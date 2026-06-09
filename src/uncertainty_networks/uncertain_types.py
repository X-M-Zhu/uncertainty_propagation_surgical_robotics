# Author: X.M. Christine Zhu
# Date: 06/03/2026

"""
uncertain_types.py — User-facing uncertain Cartesian types for CIS I.

Mirrors the nominal types (vct3, Rot, Frame) from spatial_math.py
but carries covariance. All heavy math delegates to UncertainTransform.

API (following Dr. Taylor's CIS I specification):

    uVector(v, cov)         uncertain vector, any dimension
    uPoint(p, cov)          uncertain 3-D point
    uRot(R, cov)            uncertain rotation
    uFrame(F, cov)          uncertain rigid frame

Supported operations::

    # uPoint
    up3 = up1 + up2
    up3 = p1  + up2

    # uRot applied to point
    up2 = uR * up1
    up2 = uR * p1

    # uRot composition
    uR3 = uR1 * uR2
    uR3 = R1  * uR2
    uR3 = uR1 * R2

    # uFrame from uncertain parts
    uF = uFrame(uR, up)
    uF = uFrame(R,  up)
    uF = uFrame(uR, p)

    # uFrame composition / point transform
    uF3 = uF1 * uF2
    uF3 = R1  * uF2    (Rot from the left;   also accepts Frame)
    uF3 = uF1 * R2     (Rot from the right;  also accepts Frame)
    up2 = uF  * up1
    up2 = uF  * p1

    # uRot from uncertain angle or axis
    uR = uRot(axis_str, uAngle)     uAngle is a uVector of dim 1
    uR = uRot(uAxis,    angle)      uAxis  is a uVector or uPoint
    uR = uRot(uAxis,    uAngle)     both uncertain
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as _ScipyRot

from .nominal_types import vct3, Rot, Frame
from .uncertain_geometry import UncertainTransform
from .se3 import skew


# ── helpers ───────────────────────────────────────────────────────────────────

def _axis_angle_to_rot(axis: np.ndarray, angle: float) -> Rot:
    rotvec = axis * angle
    return Rot(matrix=_ScipyRot.from_rotvec(rotvec).as_matrix())


_AXIS_VECS = {
    'x': np.array([1., 0., 0.]),
    'y': np.array([0., 1., 0.]),
    'z': np.array([0., 0., 1.]),
}


# ── uVector ───────────────────────────────────────────────────────────────────

class uVector:
    """
    Uncertain vector of any dimension n.

        uv = uVector(v)           zero covariance
        uv = uVector(v, cov)      explicit n×n covariance
    """

    def __init__(self, v, C=None):
        self.v = np.asarray(v, dtype=float).ravel()
        n = len(self.v)
        self.C = np.zeros((n, n), dtype=float) if C is None else np.asarray(C, dtype=float)

    def __repr__(self):
        return f"uVector(v={self.v}, C_diag={np.diag(self.C)})"


# ── uPoint ────────────────────────────────────────────────────────────────────

class uPoint:
    """
    Uncertain 3-D point.

        up = uPoint(p)              vct3, zero covariance
        up = uPoint(p, cov)         vct3 + 3×3 covariance
        up = uPoint([x,y,z], cov)   array-like also accepted
    """

    def __init__(self, p, C=None):
        if isinstance(p, vct3):
            self._p = p
        else:
            arr = np.asarray(p, dtype=float).ravel()
            self._p = vct3(arr[0], arr[1], arr[2])
        self.C = np.zeros((3, 3), dtype=float) if C is None else np.asarray(C, dtype=float)

    # ── properties ──
    @property
    def p(self) -> vct3:
        return self._p

    @property
    def x(self) -> float:
        return self._p.x

    @property
    def y(self) -> float:
        return self._p.y

    @property
    def z(self) -> float:
        return self._p.z

    # ── addition ──
    def __add__(self, other):
        if isinstance(other, uPoint):
            return uPoint(self._p + other._p, self.C + other.C)
        if isinstance(other, vct3):
            return uPoint(self._p + other, self.C.copy())
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, vct3):
            return uPoint(other + self._p, self.C.copy())
        return NotImplemented

    def __repr__(self):
        return (f"uPoint(p=({self.x:.4f}, {self.y:.4f}, {self.z:.4f}), "
                f"C_diag={np.diag(self.C)})")


# ── uRot ──────────────────────────────────────────────────────────────────────

class uRot:
    """
    Uncertain rotation.

    Constructors::

        uR = uRot(R)                        Rot, zero covariance
        uR = uRot(R, Calpha)                Rot + 3×3 rotation covariance
        uR = uRot(axis_str, uAngle)         axis='x'/'y'/'z', uncertain angle
        uR = uRot(uAxis, angle)             uncertain axis (uVector/uPoint), certain angle
        uR = uRot(uAxis, uAngle)            both uncertain
    """

    def __init__(self, first, second=None):
        if isinstance(first, Rot):
            self._R = first
            self.C = np.zeros((3, 3), dtype=float) if second is None \
                else np.asarray(second, dtype=float)

        elif isinstance(first, str):
            # uRot('z', uAngle)  or  uRot('z', angle_float)
            ax = _AXIS_VECS[first.lower()]
            if isinstance(second, uVector):
                angle_nom = float(second.v[0])
                self._R = Rot(axis=first, angle=angle_nom)
                self.C = np.outer(ax, ax) * float(second.C[0, 0])
            else:
                self._R = Rot(axis=first, angle=float(second))
                self.C = np.zeros((3, 3), dtype=float)

        elif isinstance(first, (uVector, uPoint)):
            # uRot(uAxis, angle)  or  uRot(uAxis, uAngle)
            if isinstance(first, uPoint):
                ax_nom = np.array([first.x, first.y, first.z])
                C_axis = first.C
            else:
                ax_nom = first.v[:3]
                C_axis = first.C[:3, :3]

            norm = np.linalg.norm(ax_nom)
            ax_nom = ax_nom / (norm + 1e-30)

            if isinstance(second, uVector):
                angle_nom = float(second.v[0])
                var_angle = float(second.C[0, 0])
                # Calpha = angle^2 * C_axis + (ax ⊗ ax) * var_angle  (first-order)
                self.C = angle_nom ** 2 * C_axis + np.outer(ax_nom, ax_nom) * var_angle
            else:
                angle_nom = float(second)
                self.C = angle_nom ** 2 * C_axis

            self._R = _axis_angle_to_rot(ax_nom, angle_nom)

        else:
            raise TypeError(
                f"uRot: cannot construct from ({type(first).__name__}, {type(second).__name__}). "
                "Expected (Rot, cov?), ('axis', uAngle), or (uAxis, angle_or_uAngle)."
            )

    # ── properties ──
    @property
    def R(self) -> Rot:
        return self._R

    @property
    def matrix(self) -> np.ndarray:
        return self._R.matrix

    # ── multiplication ──
    def __mul__(self, other):
        R_mat = self._R.matrix

        if isinstance(other, uPoint):
            p_arr = np.array([other.x, other.y, other.z])
            p_nom = R_mat @ p_arr
            J_alpha = -R_mat @ skew(p_arr)
            C_out = J_alpha @ self.C @ J_alpha.T + R_mat @ other.C @ R_mat.T
            return uPoint(vct3(*p_nom), 0.5 * (C_out + C_out.T))

        if isinstance(other, vct3):
            p_arr = np.array([other.x, other.y, other.z])
            p_nom = R_mat @ p_arr
            J_alpha = -R_mat @ skew(p_arr)
            C_out = J_alpha @ self.C @ J_alpha.T
            return uPoint(vct3(*p_nom), 0.5 * (C_out + C_out.T))

        if isinstance(other, uRot):
            R2 = other._R.matrix
            R3 = R_mat @ R2
            # α3 ≈ R2^T α1 + α2  (right-perturbation, independent)
            C3 = R2.T @ self.C @ R2 + other.C
            return uRot(Rot(matrix=R3), 0.5 * (C3 + C3.T))

        if isinstance(other, Rot):
            R2 = other.matrix
            R3 = R_mat @ R2
            # α3 ≈ R2^T α1  (R2 is certain)
            C3 = R2.T @ self.C @ R2
            return uRot(Rot(matrix=R3), 0.5 * (C3 + C3.T))

        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Rot):
            # R1 * uR2  →  R1 certain, α3 = α2
            R3 = other.matrix @ self._R.matrix
            return uRot(Rot(matrix=R3), self.C.copy())
        return NotImplemented

    # ── convention bridge ──
    @classmethod
    def from_left_perturbation(cls, R: Rot, C_left: np.ndarray) -> "uRot":
        """Construct a uRot from a left-convention (world-frame) rotation covariance.

        Left model:  R_true = Exp(alpha_L) * R_nom
        Right model: R_true = R_nom * Exp(alpha_R)

        Relationship (first-order):  alpha_L = R_nom @ alpha_R
            => C_left  = R_nom @ C_right @ R_nom^T
            => C_right = R_nom^T @ C_left @ R_nom
        """
        C_left = np.asarray(C_left, dtype=float)
        R_mat = R.matrix
        C_right = R_mat.T @ C_left @ R_mat
        return cls(R, 0.5 * (C_right + C_right.T))

    def to_right_perturbation(self) -> "uRot":
        """No-op — covariance is already in right (body-frame) convention."""
        return uRot(self._R, self.C.copy())

    def to_left_perturbation(self) -> "uRot":
        """Convert right-convention (body-frame) covariance to left-convention (world-frame).

        C_left = R_nom @ C_right @ R_nom^T
        """
        R_mat = self._R.matrix
        C_left = R_mat @ self.C @ R_mat.T
        return uRot(self._R, 0.5 * (C_left + C_left.T))

    def __repr__(self):
        return f"uRot(C_diag={np.diag(self.C)})"


# ── uFrame ────────────────────────────────────────────────────────────────────

class uFrame:
    """
    Uncertain rigid frame.

    Constructors::

        uF = uFrame(F)              Frame, zero covariance
        uF = uFrame(F, covEta)      Frame + 6×6 pose covariance
        uF = uFrame(uR, up)         uncertain rotation + uncertain point
        uF = uFrame(R,  up)         certain  rotation + uncertain point
        uF = uFrame(uR, p)          uncertain rotation + certain  point

    Operations::

        uF3 = uF1 * uF2    frame composition (full uncertainty propagation)
        uF3 = F1  * uF2    certain left frame
        uF3 = uF1 * F2     certain right frame
        up2 = uF  * up1    transform uncertain point
        up2 = uF  * p1     transform certain point
    """

    def __init__(self, first, second=None):
        if isinstance(first, UncertainTransform):
            self._ut = first

        elif isinstance(first, Frame):
            F_nom = np.eye(4, dtype=float)
            F_nom[:3, :3] = first.R.matrix
            F_nom[:3, 3] = [first.p.x, first.p.y, first.p.z]
            C = np.zeros((6, 6), dtype=float) if second is None \
                else np.asarray(second, dtype=float)
            self._ut = UncertainTransform(F_nom, C)

        elif isinstance(first, uRot) and isinstance(second, uPoint):
            # uFrame(uR, up)
            F_nom = np.eye(4, dtype=float)
            F_nom[:3, :3] = first.R.matrix
            F_nom[:3, 3] = [second.x, second.y, second.z]
            C = np.zeros((6, 6), dtype=float)
            C[:3, :3] = first.C   # rotation covariance  (alpha)
            C[3:, 3:] = second.C  # translation covariance (epsilon)
            self._ut = UncertainTransform(F_nom, C)

        elif isinstance(first, Rot) and isinstance(second, uPoint):
            # uFrame(R, up)
            F_nom = np.eye(4, dtype=float)
            F_nom[:3, :3] = first.matrix
            F_nom[:3, 3] = [second.x, second.y, second.z]
            C = np.zeros((6, 6), dtype=float)
            C[3:, 3:] = second.C
            self._ut = UncertainTransform(F_nom, C)

        elif isinstance(first, uRot) and isinstance(second, vct3):
            # uFrame(uR, p)
            F_nom = np.eye(4, dtype=float)
            F_nom[:3, :3] = first.R.matrix
            F_nom[:3, 3] = [second.x, second.y, second.z]
            C = np.zeros((6, 6), dtype=float)
            C[:3, :3] = first.C
            self._ut = UncertainTransform(F_nom, C)

        else:
            raise TypeError(
                f"uFrame: unsupported arguments ({type(first).__name__}, {type(second).__name__}). "
                "Expected (Frame, cov?), (uRot, uPoint), (Rot, uPoint), or (uRot, vct3)."
            )

    # ── properties ──
    @property
    def F(self) -> Frame:
        return self._ut.to_frame()

    @property
    def F_nom(self) -> np.ndarray:
        return self._ut.F_nom

    @property
    def C(self) -> np.ndarray:
        return self._ut.C

    def inv(self) -> "uFrame":
        return uFrame(self._ut.inv())

    def to_uncertain_transform(self) -> UncertainTransform:
        return self._ut

    # ── convention bridge ──
    @classmethod
    def from_left_perturbation(cls, frame_or_F_nom, C_left: np.ndarray) -> "uFrame":
        """Construct a uFrame from a left-convention (world-frame) pose covariance.

        Left model:  T_true = Exp(eta_L) * F_nom
        Right model: T_true = F_nom * Exp(eta_R)

        Relationship (first-order):  eta_L = Ad_{F_nom} @ eta_R
            => C_right = Ad_{F_nom^{-1}} @ C_left @ Ad_{F_nom^{-1}}^T

        Accepts either a Frame object or a raw 4×4 numpy array as the nominal transform.
        """
        if isinstance(frame_or_F_nom, Frame):
            F_nom = np.eye(4, dtype=float)
            F_nom[:3, :3] = frame_or_F_nom.R.matrix
            F_nom[:3, 3] = [frame_or_F_nom.p.x, frame_or_F_nom.p.y, frame_or_F_nom.p.z]
        else:
            F_nom = np.asarray(frame_or_F_nom, dtype=float)
        return cls(UncertainTransform.from_left_perturbation(F_nom, C_left))

    def to_right_perturbation(self) -> "uFrame":
        """No-op — covariance is already in right (body-frame) convention."""
        return uFrame(self._ut.to_right_perturbation())

    def to_left_perturbation(self) -> "uFrame":
        """Convert right-convention (body-frame) covariance to left-convention (world-frame).

        C_left = Ad_{F_nom} @ C_right @ Ad_{F_nom}^T
        """
        return uFrame(self._ut.to_left_perturbation())

    # ── multiplication ──
    def __mul__(self, other):
        if isinstance(other, uFrame):
            return uFrame(self._ut.compose(other._ut))

        if isinstance(other, Frame):
            F_nom = np.eye(4, dtype=float)
            F_nom[:3, :3] = other.R.matrix
            F_nom[:3, 3] = [other.p.x, other.p.y, other.p.z]
            return uFrame(self._ut.compose(UncertainTransform(F_nom, np.zeros((6, 6)))))

        if isinstance(other, Rot):
            F_nom = np.eye(4, dtype=float)
            F_nom[:3, :3] = other.matrix
            return uFrame(self._ut.compose(UncertainTransform(F_nom, np.zeros((6, 6)))))

        if isinstance(other, uPoint):
            p_out, C_out = self._ut.transform_point(other.p, Cp=other.C)
            return uPoint(p_out, C_out)

        if isinstance(other, vct3):
            p_out, C_out = self._ut.transform_point(other)
            return uPoint(p_out, C_out)

        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Frame):
            F_nom = np.eye(4, dtype=float)
            F_nom[:3, :3] = other.R.matrix
            F_nom[:3, 3] = [other.p.x, other.p.y, other.p.z]
            return uFrame(UncertainTransform(F_nom, np.zeros((6, 6))).compose(self._ut))
        if isinstance(other, Rot):
            F_nom = np.eye(4, dtype=float)
            F_nom[:3, :3] = other.matrix
            return uFrame(UncertainTransform(F_nom, np.zeros((6, 6))).compose(self._ut))
        return NotImplemented

    def __matmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        p = self._ut.F_nom[:3, 3]
        return (f"uFrame(p=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}), "
                f"C_diag={np.diag(self._ut.C)})")
