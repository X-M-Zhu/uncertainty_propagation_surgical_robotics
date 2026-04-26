# Author: X.M. Christine Zhu
# Date: 02/06/2026

"""
This module implements the core uncertain geometric primitives described in:
  docs/math_note.pdf

Scope:
  - Forward uncertainty propagation on SE(3) using first-order approximations
  - CIS I left-multiplicative perturbation convention
  - No estimation / filtering / optimization

Convention (CIS I):
  - Nominal transform: F_nom ∈ SE(3) (4×4 homogeneous matrix)
  - Pose perturbation: eta = [alpha; epsilon] ∈ R^6,  eta ~ N(0, C)
      alpha   ∈ R^3 rotation perturbation
      epsilon ∈ R^3 translation perturbation
  - Left perturbation model:
      T_true = Exp(eta) ∘ F_nom

Core propagation rule (independent edges):
  If F_ab = {F_nom,ab, C_ab} and F_bc = {F_nom,bc, C_bc}, then

      F_nom,ac = F_nom,ab ∘ F_nom,bc
      C_ac ≈ C_ab + Ad_{F_nom,ab} C_bc Ad_{F_nom,ab}^T

where Ad_T is the SE(3) adjoint under CIS I twist ordering [alpha; epsilon].
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .se3 import adjoint_se3, inv_se3, is_se3, skew

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

    Perturbation model (CIS I, left-multiplicative):
        T_true = Exp(eta) ∘ F_nom
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

    def inv(self) -> "UncertainTransform":
        r"""
        Invert an uncertain transform (first-order).

        Nominal inverse:
            F_nom^{-1} = inv_se3(F_nom)

        Covariance mapping:
            Under the CIS I left-perturbation model, the inverse perturbation is
            mapped by the adjoint of the inverse nominal transform. Using a
            first-order approximation:

                C_inv ≈ Ad_{F_nom^{-1}} C Ad_{F_nom^{-1}}^T

        Returns
        -------
        UncertainTransform
            Inverse uncertain transform.
        """
        F_inv = inv_se3(self.F_nom)
        Ad_Finv = adjoint_se3(F_inv)
        C_inv = Ad_Finv @ self.C @ Ad_Finv.T
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
            C_ac ≈ C_ab + Ad_{F_nom,ab} C_bc Ad_{F_nom,ab}^T

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

        Ad_Fab = adjoint_se3(F_ab)

        # Current scope: independent edges; cross-covariances not tracked
        if assume_independent:
            C_ac = self.C + Ad_Fab @ other.C @ Ad_Fab.T
        else:
            C_ac = self.C + Ad_Fab @ other.C @ Ad_Fab.T

        return UncertainTransform(F_ac, C_ac)

    def __matmul__(self, other: "UncertainTransform") -> "UncertainTransform":
        r"""
        Operator overload for composition:
            F_ac = F_ab @ F_bc
        """
        return self.compose(other)

    def transform_point(self, p: Array, Cp: Array | None = None) -> tuple[Array, Array]:
        r"""
        Transform a 3D point and propagate uncertainty using CIS I Jacobians.

        Nominal point transform:
            p'_nom = R p + t

        CIS I left-perturbation linearization:
            If T_true = Exp(eta) ∘ T_nom with eta = [alpha; epsilon],
            then to first order:

                δp' ≈ -[p'_nom]× alpha + epsilon

        Therefore, the Jacobians are:
            J_eta = [ -[p'_nom]×   I_3 ]    (shape 3×6)
            J_p   = R                       (shape 3×3)

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
        p = np.asarray(p, dtype=float).reshape(3)

        R = self.F_nom[:3, :3]
        t = self.F_nom[:3, 3]

        # Nominal transformation
        p_nom = R @ p + t

        # CIS I Jacobian w.r.t. pose perturbation eta = [alpha; epsilon]
        J_eta = np.zeros((3, 6), dtype=float)
        J_eta[:, :3] = -skew(p_nom)       # d p' / d alpha
        J_eta[:, 3:] = np.eye(3, dtype=float)  # d p' / d epsilon

        Cp_pose = J_eta @ self.C @ J_eta.T

        if Cp is None:
            Cp_out = Cp_pose
        else:
            Cp = np.asarray(Cp, dtype=float).reshape(3, 3)
            Cp_point = R @ Cp @ R.T
            Cp_out = Cp_pose + Cp_point

        # Defensive symmetrization
        Cp_out = 0.5 * (Cp_out + Cp_out.T)
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
