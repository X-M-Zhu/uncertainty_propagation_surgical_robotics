from dataclasses import dataclass
import math
import numpy as np
from src.uncertainty_networks.spatial import Point, Rot, Frame 

class uPoint:
    """An uncertain 3D position point."""
    def __init__(self, p: Point, covEpsilon=None):
        if not isinstance(p, Point):
            raise TypeError("p must be an instance of the nominal Point class")
        self.p = p
        
        if covEpsilon is None:
            self.cov = np.zeros((3, 3))
        else:
            self.cov = np.asarray(covEpsilon, dtype=float)
            if self.cov.shape != (3, 3):
                raise ValueError("covEpsilon must be a 3x3 matrix.")

    def __add__(self, other):
        """Supports: up3 = up1 + up2 and up3 = p1 + up2."""
        if isinstance(other, uPoint):
            # Covariances add directly for independent variables
            return uPoint(self.p + other.p, self.cov + other.cov)
        elif isinstance(other, Point):
            # Adding a nominal Point (cov = 0)
            return uPoint(self.p + other, self.cov)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __repr__(self):
        return f"uPoint(p={self.p.tolist()}, cov_eps=\n{self.cov})"

class uVec:
    """An uncertain vector of arbitrary dimension."""
    def __init__(self, v, covDeltav=None):
        self.v = np.asarray(v, dtype=float).flatten()
        if covDeltav is None:
            self.cov = np.zeros((self.v.size, self.v.size))
        else:
            self.cov = np.asarray(covDeltav, dtype=float)
            if self.cov.shape != (self.v.size, self.v.size):
                raise ValueError("Covariance shape must match vector dimension.")

    def __repr__(self):
        return f"uVector(\n  v={self.v},\n  cov=\n{self.cov}\n)"

class uRot:
    """An uncertain 3D Rotation Matrix with small perturbation angles alpha."""
    def __init__(self, R: Rot, covAlpha=None):
        if not isinstance(R, Rot):
            raise TypeError("R must be an instance of the nominal Rot class")
        self.R = R
            
        if covAlpha is None:
            self.cov = np.zeros((3, 3))
        else:
            self.cov = np.asarray(covAlpha, dtype=float)
            if self.cov.shape != (3, 3):
                raise ValueError("covAlpha must be a 3x3 matrix.")

    def inv(self) -> 'uRot':
        """Computes the inverse rotation and propagates its alpha covariance matrix."""
        pass

    def __mul__(self, other):
        """Supports:
        uR3 = uR1 * uR2 (Rotation Composition)
        uR3 = R1 * uR2
        uR3 = uR1 * R2
        up2 = uR * up1 (Rotating an uncertain point)
        up2 = uR * p1  (Rotating a certain nominal point)
        """
        if isinstance(other, uRot):
            # alpha_3 = R2^T * alpha_1 + alpha_2
            R3 = self.R * other.R  # nominal multiplication
            R2_inv = other.R.matrix.inv()
            cov3 = (R2_inv @ self.cov @ other.R.matrix) + other.cov
            return uRot(R3, cov3)
            
        elif isinstance(other, Rot):
            # Multiplying uRot * Rot (other has no covariance)
            R3 = self.R * other
            R2_T = other.matrix.inv()
            cov3 = R2_T @ self.cov @ other.matrix
            return uRot(R3, cov3)

        elif isinstance(other, uPoint):
            pass
            # Form: p2 = R1 * p1
            # Jacobian w.r.t alpha error from slide 39: -R * skew(p1) [cite: 1010, 1022]
            p2 = self.R * other.p  # Reuse nominal operator
            J_alpha = -self.R.matrix @ skew(other.p.vec)
            cov2 = (self.R.matrix @ other.cov @ self.R.matrix.T) + (J_alpha @ self.cov @ J_alpha.T)
            return uPoint(p2, cov2)
            
        elif isinstance(other, Point):
            pass
            # Form: p2 = uR * p1 (where p1 is certain nominal Point)
            p2 = self.R * other
            J_alpha = -self.R.matrix @ skew(other.vec)
            cov2 = J_alpha @ self.cov @ J_alpha.T
            return uPoint(p2, cov2)
            
        return NotImplemented

    def __rmul__(self, other):
        # Supports: R1 * uR2
        if isinstance(other, Rot):
            R3 = other * self.R
            # Because the certain rotation is on the left, it doesn't affect the right-side alpha frame
            return uRot(R3, self.cov)
        return NotImplemented

    def __repr__(self):
        return f"uRot(\n  nominal matrix=\n{self.R.matrix},\n  cov_alpha=\n{self.cov}\n)"