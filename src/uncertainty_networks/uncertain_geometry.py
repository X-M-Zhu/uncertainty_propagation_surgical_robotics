# Author: X.M. Christine Zhu
# Date: 02/06/2026

"""
This module implements the core uncertain geometric primitives described in:
  docs/math_note.pdf

Scope:
  - Forward uncertainty propagation on SE(3) using first-order approximations
  - CIS I right-multiplicative perturbation convention
  - No estimation / filtering / optimization

Convention (CIS I):
  - Nominal transform: F_nom ∈ SE(3) (4×4 homogeneous matrix)
  - Pose perturbation: eta = [alpha; epsilon] ∈ R^6,  eta ~ N(0, C)
      alpha   ∈ R^3 rotation perturbation
      epsilon ∈ R^3 translation perturbation
  - Right perturbation model:
      T_true = F_nom ∘ Exp(eta)

Core propagation rule (independent edges):
  If F_ab = {F_nom,ab, C_ab} and F_bc = {F_nom,bc, C_bc}, then

      F_nom,ac = F_nom,ab ∘ F_nom,bc
      C_ac ≈ Ad_{F_nom,bc^{-1}} C_ab Ad_{F_nom,bc^{-1}}^T + C_bc

where Ad_T is the SE(3) adjoint under CIS I twist ordering [alpha; epsilon].
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .se3 import adjoint_se3, inv_se3, is_se3, skew
from .nominal_types import vct3, Rot, Frame

Array = np.ndarray


@dataclass(frozen=True)
class UncertainTransform:
    r"""
    Uncertain rigid-body transformation in SE(3).

    Representation:
        F = {F_nom, C}

    where:
        - F_nom ∈ SE(3) is the nominal 4×4 homogeneous transform
        - C ∈ R^{6×6} is the covariance of the pose perturbation eta

    Perturbation model (CIS I, right-multiplicative):
        T_true = F_nom ∘ Exp(eta)
        eta = [alpha; epsilon] ~ N(0, C)

    This class supports:
        - composition with first-order uncertainty propagation
        - inversion with first-order covariance mapping
        - point transformation with CIS I Jacobian (implemented later in this file)
    """
    F_nom: Array
    C: Array

    @staticmethod
    def identity(C: Array | None = None) -> "UncertainTransform":
        r"""
        Construct the identity transform with optional covariance.

        Identity:
            F_nom = I_4

        If C is not provided, the covariance defaults to zero.

        Parameters
        ----------
        C : ndarray, optional, shape (6,6)
            Covariance for the identity transform.

        Returns
        -------
        UncertainTransform
            Identity uncertain transform.
        """
        if C is None:
            C = np.zeros((6, 6), dtype=float)
        return UncertainTransform(np.eye(4, dtype=float), C)

    @classmethod
    def from_frame(cls, frame: Frame, C: np.ndarray | None = None) -> "UncertainTransform":
        r"""
        Construct an UncertainTransform from a spatial_math.Frame object.

        Parameters
        ----------
        frame : Frame
            Nominal rigid-body frame (rotation + translation).
        C : ndarray, optional, shape (6,6)
            Covariance of the pose perturbation. Defaults to zero.

        Returns
        -------
        UncertainTransform
        """
        F = np.eye(4, dtype=float)
        F[:3, :3] = frame.R.matrix
        F[:3, 3] = np.array([frame.p.x, frame.p.y, frame.p.z])
        if C is None:
            C = np.zeros((6, 6), dtype=float)
        return cls(F, C)

    def to_frame(self) -> Frame:
        r"""
        Convert the nominal transform to a spatial_math.Frame object.

        Returns
        -------
        Frame
            Frame with the same rotation and translation as F_nom.
        """
        return Frame(
            R=Rot(matrix=self.F_nom[:3, :3]),
            p=vct3(float(self.F_nom[0, 3]),
                   float(self.F_nom[1, 3]),
                   float(self.F_nom[2, 3])),
        )

    def inv(self) -> "UncertainTransform":
        r"""
        Invert an uncertain transform (first-order).

        Nominal inverse:
            F_nom^{-1} = inv_se3(F_nom)

        Covariance mapping:
            Under the CIS I right-perturbation model, the inverse perturbation is
            mapped by the adjoint of the nominal transform. Using a
            first-order approximation:

                C_inv ≈ Ad_{F_nom} C Ad_{F_nom}^T

        Returns
        -------
        UncertainTransform
            Inverse uncertain transform.
        """
        F_inv = inv_se3(self.F_nom)
        Ad_F = adjoint_se3(self.F_nom)
        C_inv = Ad_F @ self.C @ Ad_F.T
        return UncertainTransform(F_inv, C_inv)

    def compose(self, other: "UncertainTransform", assume_independent: bool = True) -> "UncertainTransform":
        r"""
        Compose two uncertain transforms (first-order propagation).

        Let:
            self  = F_ab = {F_nom,ab, C_ab}
            other = F_bc = {F_nom,bc, C_bc}

        Nominal composition:
            F_nom,ac = F_nom,ab ∘ F_nom,bc  (matrix product)

        First-order covariance propagation (independent edges):
            C_ac ≈ Ad_{F_nom,bc^{-1}} C_ab Ad_{F_nom,bc^{-1}}^T + C_bc

        This is the core propagation rule used throughout the framework.

        Parameters
        ----------
        other : UncertainTransform
            The transform to compose on the right.
        assume_independent : bool
            If True, assumes perturbations are independent (default).
            (Cross-covariances are not tracked in the current scope.)

        Returns
        -------
        UncertainTransform
            Composed uncertain transform.
        """
        F_ab = self.F_nom
        F_bc = other.F_nom
        F_ac = F_ab @ F_bc

        Ad_Fbc_inv = adjoint_se3(inv_se3(F_bc))

        # Current scope: independent edges; cross-covariances not tracked
        if assume_independent:
            C_ac = Ad_Fbc_inv @ self.C @ Ad_Fbc_inv.T + other.C
        else:
            C_ac = Ad_Fbc_inv @ self.C @ Ad_Fbc_inv.T + other.C

        return UncertainTransform(F_ac, C_ac)

    def __matmul__(self, other: "UncertainTransform") -> "UncertainTransform":
        r"""
        Operator overload for composition:
            F_ac = F_ab @ F_bc
        """
        return self.compose(other)

    @classmethod
    def from_left_perturbation(cls, F_nom: Array, C_left: Array) -> "UncertainTransform":
        r"""
        Construct an UncertainTransform from a *left*-perturbation covariance.

        Left model:   T_true = Exp(eta_L) * F_nom
        Right model:  T_true = F_nom      * Exp(eta_R)

        Relationship:
            eta_L = Ad_{F_nom} * eta_R
            =>  C_left  = Ad_{F_nom}     * C_right * Ad_{F_nom}^T
            =>  C_right = Ad_{F_nom^{-1}} * C_left  * Ad_{F_nom^{-1}}^T

        Use this when an external source (another library, a sensor driver,
        a paper that uses the world-frame convention) provides a covariance in
        the left-perturbation convention and you need to feed it into this
        right-perturbation framework.

        Parameters
        ----------
        F_nom : array-like, shape (4,4)
            Nominal SE(3) transform.
        C_left : array-like, shape (6,6)
            Covariance expressed in the *left*-perturbation convention.

        Returns
        -------
        UncertainTransform
            Same nominal transform, covariance converted to right-perturbation.
        """
        F_nom  = np.asarray(F_nom,  dtype=float)
        C_left = np.asarray(C_left, dtype=float)
        Ad_inv = adjoint_se3(inv_se3(F_nom))
        C_right = Ad_inv @ C_left @ Ad_inv.T
        return cls(F_nom, C_right)

    def to_right_perturbation(self) -> "UncertainTransform":
        r"""
        Return this transform — covariance is already in right-perturbation convention.

        Right model:  T_true = F_nom * Exp(eta_R)

        Since this class natively uses right perturbation, this is a no-op.

        Returns
        -------
        UncertainTransform
            Same uncertain transform (no-op).
        """
        return UncertainTransform(self.F_nom, self.C.copy())

    def to_left_perturbation(self) -> "UncertainTransform":
        r"""
        Convert this right-perturbation covariance to its left-perturbation equivalent.

        This library uses right-perturbation (body-frame) as the default:
            T_true = F_nom * Exp(eta_R)        (right / body-frame convention)

        The equivalent left-perturbation (world-frame) covariance satisfies:
            T_true = Exp(eta_L) * F_nom        (left / world-frame convention)

        Relationship:
            eta_L = Ad_{F_nom} * eta_R
            =>  C_left = Ad_{F_nom} * C_right * Ad_{F_nom}^T

        Returns
        -------
        UncertainTransform
            Same nominal transform, covariance expressed in left-perturbation convention.
        """
        Ad = adjoint_se3(self.F_nom)
        C_left = Ad @ self.C @ Ad.T
        return UncertainTransform(self.F_nom, C_left)

    def transform_point(self, p: Array, Cp: Array | None = None) -> tuple[Array, Array]:
        r"""
        Transform a 3D point and propagate uncertainty using CIS I Jacobians.

        Nominal point transform:
            p'_nom = R p + t

        CIS I right-perturbation linearization:
            If T_true = F_nom ∘ Exp(eta) with eta = [alpha; epsilon],
            then to first order:

                δp' ≈ -R [p]× alpha + R epsilon

        Therefore, the Jacobians are:
            J_eta = [ -R [p]×   R ]    (shape 3×6)
            J_p   = R              (shape 3×3)

        where p is the **input** point (in the source frame).

        Covariance propagation:
            If point has intrinsic covariance Cp (in the input point's frame),
            then:

                Cp' ≈ J_eta C J_eta^T + R Cp R^T

        If Cp is None, we return the pose-induced covariance only:
                Cp' ≈ J_eta C J_eta^T

        Parameters
        ----------
        p : array-like, shape (3,)
            Input point (3D).
        Cp : ndarray, optional, shape (3,3)
            Intrinsic point covariance.

        Returns
        -------
        p_nom : ndarray, shape (3,)
            Nominal transformed point.
        Cp_out : ndarray, shape (3,3)
            Propagated covariance of transformed point.
        """
        return_point = isinstance(p, vct3)
        if return_point:
            p_arr = np.array([p.x, p.y, p.z], dtype=float)
        else:
            p_arr = np.asarray(p, dtype=float).reshape(3)

        R = self.F_nom[:3, :3]
        t = self.F_nom[:3, 3]

        # Nominal transformation
        p_nom = R @ p_arr + t

        # CIS I Jacobian w.r.t. pose perturbation eta = [alpha; epsilon]
        # Right convention: d p' / d eta = [-R skew(p_in), R]
        J_eta = np.zeros((3, 6), dtype=float)
        J_eta[:, :3] = -R @ skew(p_arr)     # d p' / d alpha (input point)
        J_eta[:, 3:] = R                    # d p' / d epsilon

        Cp_pose = J_eta @ self.C @ J_eta.T

        if Cp is None:
            Cp_out = Cp_pose
        else:
            Cp = np.asarray(Cp, dtype=float).reshape(3, 3)
            Cp_point = R @ Cp @ R.T
            Cp_out = Cp_pose + Cp_point

        # Defensive symmetrization
        Cp_out = 0.5 * (Cp_out + Cp_out.T)

        if return_point:
            return vct3(float(p_nom[0]), float(p_nom[1]), float(p_nom[2])), Cp_out
        return p_nom, Cp_out

    def __post_init__(self) -> None:
        F = np.asarray(self.F_nom, dtype=float)
        C = np.asarray(self.C, dtype=float)

        if F.shape != (4, 4):
            raise ValueError(f"F_nom must be shape (4,4), got {F.shape}")
        if C.shape != (6, 6):
            raise ValueError(f"C must be shape (6,6), got {C.shape}")
        if not is_se3(F):
            raise ValueError("F_nom does not appear to be a valid SE(3) homogeneous transform.")

        # Defensively symmetrize covariance (numerical stability)
        C = 0.5 * (C + C.T)

        object.__setattr__(self, "F_nom", F)
        object.__setattr__(self, "C", C)
