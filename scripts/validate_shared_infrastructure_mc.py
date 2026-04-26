# Author: X.M. Christine Zhu
# Date: 02/06/2026

import numpy as np

from uncertainty_networks import UncertainTransform, GeometricNetwork
from uncertainty_networks.se3 import make_se3, rotz, exp_se3


def cov_sample(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / (X.shape[0] - 1)


def rel_frob(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A - B, ord="fro") / np.linalg.norm(B, ord="fro"))


def main():
    seed = 12
    N = 80000
    rng = np.random.default_rng(seed)

    net = GeometricNetwork()

    # ------------------------------------------------------------------
    # Shared infrastructure frames (all connect to World W)
    # ------------------------------------------------------------------
    # World frame
    net.add_frame("W")

    # Tracker system: W <- Trk
    C_trk = np.diag([4e-6] * 6)
    T_w_trk = make_se3(rotz(0.02), [0.02, 0.00, 0.01])
    U_w_trk = UncertainTransform(T_w_trk, C_trk)
    net.add_edge("Trk", "W", U_w_trk, add_inverse=True)

    # CT system: W <- CT
    C_ct = np.diag([5e-6] * 6)
    T_w_ct = make_se3(rotz(-0.015), [0.03, -0.01, 0.00])
    U_w_ct = UncertainTransform(T_w_ct, C_ct)
    net.add_edge("CT", "W", U_w_ct, add_inverse=True)

    # Robot base system: W <- Rb
    C_rb = np.diag([6e-6] * 6)
    T_w_rb = make_se3(rotz(0.01), [-0.02, 0.02, 0.00])
    U_w_rb = UncertainTransform(T_w_rb, C_rb)
    net.add_edge("Rb", "W", U_w_rb, add_inverse=True)

    # ------------------------------------------------------------------
    # Branches under each system
    # ------------------------------------------------------------------
    # Tracker branch: Trk -> Cam -> MarkerBody
    C_cam = np.diag([3e-6] * 6)
    T_trk_cam = make_se3(rotz(0.03), [0.10, 0.00, 0.00])
    U_trk_cam = UncertainTransform(T_trk_cam, C_cam)
    net.add_edge("Trk", "Cam", U_trk_cam, add_inverse=True)

    C_mk = np.diag([2e-6] * 6)
    T_cam_mk = make_se3(rotz(-0.02), [0.05, 0.02, 0.00])
    U_cam_mk = UncertainTransform(T_cam_mk, C_mk)
    net.add_edge("Cam", "Mk", U_cam_mk, add_inverse=True)

    # Robot branch: Rb -> Tool
    C_tool = np.diag([4e-6] * 6)
    T_rb_tool = make_se3(rotz(0.04), [0.00, 0.15, 0.02])
    U_rb_tool = UncertainTransform(T_rb_tool, C_tool)
    net.add_edge("Rb", "Tool", U_rb_tool, add_inverse=True)

    # CT branch: CT -> Anat
    C_anat = np.diag([3e-6] * 6)
    T_ct_anat = make_se3(rotz(0.025), [0.00, 0.00, 0.10])
    U_ct_anat = UncertainTransform(T_ct_anat, C_anat)
    net.add_edge("CT", "Anat", U_ct_anat, add_inverse=True)

    # ------------------------------------------------------------------
    # Points (intrinsic)
    # ------------------------------------------------------------------
    Cp = 2e-6 * np.eye(3)

    # A tracked marker point (attached to Mk)
    p_mk = np.array([0.02, 0.00, 0.00])
    net.add_point("p_marker", "Mk", p_mk, Cp)

    # A tool tip point (attached to Tool)
    p_tip = np.array([0.00, 0.00, -0.12])
    net.add_point("p_tip", "Tool", p_tip, Cp)

    # An anatomy landmark point (attached to Anat)
    p_land = np.array([0.03, -0.01, 0.02])
    net.add_point("p_landmark", "Anat", p_land, Cp)

    # ------------------------------------------------------------------
    # Choose a point-to-point query in World (W)
    # We’ll compare tool tip vs anatomy landmark (shares: W only, but NOT trk/cam)
    # And marker vs tool tip (shares: W, and possibly none else)
    # ------------------------------------------------------------------
    src = "p_tip"
    dst = "p_landmark"
    query_frame = "W"

    delta_ind, C_ind = net.query_relative_vector_independent(src, dst, query_frame)
    delta_corr, C_corr = net.query_relative_vector(src, dst, query_frame)

    # ------------------------------------------------------------------
    # Monte Carlo simulation:
    # Sample each EDGE independently (as assumed by the model),
    # then compute the same delta = p_dst(W) - p_src(W)
    # ------------------------------------------------------------------
    # We manually sample all uncertain edges we created above.
    # (This mirrors the same independence assumptions used in the analytic propagation.)

    mean0_6 = np.zeros(6)
    mean0_3 = np.zeros(3)

    deltas = np.zeros((N, 3), dtype=float)

    # Save nominal transforms for sampling
    nom = {
        "W_Trk": (T_w_trk, C_trk),
        "W_CT": (T_w_ct, C_ct),
        "W_Rb": (T_w_rb, C_rb),
        "Trk_Cam": (T_trk_cam, C_cam),
        "Cam_Mk": (T_cam_mk, C_mk),
        "Rb_Tool": (T_rb_tool, C_tool),
        "CT_Anat": (T_ct_anat, C_anat),
    }

    for i in range(N):
        # sample each transform
        T_w_trk_s = exp_se3(rng.multivariate_normal(mean0_6, nom["W_Trk"][1])) @ nom["W_Trk"][0]
        T_w_ct_s  = exp_se3(rng.multivariate_normal(mean0_6, nom["W_CT"][1]))  @ nom["W_CT"][0]
        T_w_rb_s  = exp_se3(rng.multivariate_normal(mean0_6, nom["W_Rb"][1]))  @ nom["W_Rb"][0]

        T_trk_cam_s = exp_se3(rng.multivariate_normal(mean0_6, nom["Trk_Cam"][1])) @ nom["Trk_Cam"][0]
        T_cam_mk_s  = exp_se3(rng.multivariate_normal(mean0_6, nom["Cam_Mk"][1]))  @ nom["Cam_Mk"][0]

        T_rb_tool_s = exp_se3(rng.multivariate_normal(mean0_6, nom["Rb_Tool"][1])) @ nom["Rb_Tool"][0]
        T_ct_anat_s = exp_se3(rng.multivariate_normal(mean0_6, nom["CT_Anat"][1])) @ nom["CT_Anat"][0]

        # intrinsic point noise
        dp_src = rng.multivariate_normal(mean0_3, Cp)
        dp_dst = rng.multivariate_normal(mean0_3, Cp)

        # compute p_tip in W:
        # Tool -> Rb -> W (we have W<-Rb and Rb->Tool so Tool->W = (W<-Rb) ∘ (Rb->Tool)
        # but our stored nom is W<-Rb and Rb->Tool; Tool->W = (W<-Rb) ∘ (Rb->Tool)
        T_w_tool = T_w_rb_s @ T_rb_tool_s
        R_w_tool = T_w_tool[:3, :3]
        t_w_tool = T_w_tool[:3, 3]
        p_src_w = R_w_tool @ (p_tip + dp_src) + t_w_tool

        # compute p_landmark in W:
        # Anat -> CT -> W = (W<-CT) ∘ (CT->Anat)
        T_w_anat = T_w_ct_s @ T_ct_anat_s
        R_w_anat = T_w_anat[:3, :3]
        t_w_anat = T_w_anat[:3, 3]
        p_dst_w = R_w_anat @ (p_land + dp_dst) + t_w_anat

        deltas[i, :] = p_dst_w - p_src_w

    C_mc = cov_sample(deltas)

    err_ind = rel_frob(C_ind, C_mc)
    err_corr = rel_frob(C_corr, C_mc)

    print("\n=== Shared infrastructure MC validation ===")
    print(f"src={src}, dst={dst}, query_frame={query_frame}")
    print(f"Seed={seed}, N={N}")
    print(f"||delta_ind - delta_corr|| = {np.linalg.norm(delta_ind - delta_corr):.3e}")
    print(f"rel frob error (independent) : {err_ind:.4f}")
    print(f"rel frob error (corr-aware)  : {err_corr:.4f}")
    print(f"trace(C_ind)  = {np.trace(C_ind):.6e}")
    print(f"trace(C_corr) = {np.trace(C_corr):.6e}")
    print(f"trace(C_mc)   = {np.trace(C_mc):.6e}")

    np.set_printoptions(precision=6, suppress=False)
    print("diag(C_mc)   :", np.diag(C_mc))
    print("diag(C_ind)  :", np.diag(C_ind))
    print("diag(C_corr) :", np.diag(C_corr))


if __name__ == "__main__":
    main()
