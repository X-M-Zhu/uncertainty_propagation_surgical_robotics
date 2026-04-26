#Author: X.M. Christine Zhu
#!/usr/bin/env python3

"""
Monte Carlo validation for frame → point uncertainty propagation.

This validates the CIS-I linearization used in

    UncertainTransform.transform_point()

We compare the analytic covariance returned by transform_point()
with covariance estimated from Monte Carlo sampling of the pose
perturbation.

Model:

    T_true = Exp(eta) ∘ F_nom
    eta ~ N(0, C)

Point transform:

    p' = R p + t

Linearization used in the code:

    δp' ≈ -[p']× alpha + epsilon
"""

import numpy as np

from uncertainty_networks.uncertain_geometry import UncertainTransform
from uncertainty_networks.se3 import exp_se3, skew


def make_random_transform():
    """Generate a random SE(3) transform."""
    R = np.eye(3)
    t = np.array([0.3, -0.2, 0.5])

    F = np.eye(4)
    F[:3, :3] = R
    F[:3, 3] = t
    return F


def main():

    np.random.seed(7)

    N = 20000

    print("\n=== Frame → Point Monte Carlo Validation ===")
    print("Seed:", 7)
    print("N samples:", N)

    # Nominal transform
    F_nom = make_random_transform()

    # Pose covariance
    sigma_rot = 0.002      # radians
    sigma_trans = 0.003    # meters

    C = np.diag([
        sigma_rot**2,
        sigma_rot**2,
        sigma_rot**2,
        sigma_trans**2,
        sigma_trans**2,
        sigma_trans**2,
    ])

    F = UncertainTransform(F_nom, C)

    # Point in frame b
    p = np.array([0.4, -0.1, 0.2])

    # Analytic propagation
    p_nom, Cp_analytic = F.transform_point(p)

    samples = []

    for _ in range(N):

        eta = np.random.multivariate_normal(np.zeros(6), C)

        T_sample = exp_se3(eta) @ F_nom

        R = T_sample[:3, :3]
        t = T_sample[:3, 3]

        p_sample = R @ p + t

        samples.append(p_sample)

    samples = np.array(samples)

    p_mc = np.mean(samples, axis=0)

    Cp_mc = np.cov(samples.T)

    err = np.linalg.norm(Cp_analytic - Cp_mc, ord="fro") / np.linalg.norm(Cp_mc, ord="fro")

    print("\nRelative Frobenius error:", f"{err:.4f}")

    print("\ndiag(Cp_analytic):", np.diag(Cp_analytic))
    print("diag(Cp_mc)      :", np.diag(Cp_mc))


if __name__ == "__main__":
    main()