# Author: X.M. Christine Zhu
# Date: 02/06/2026

"""
Random-network Monte Carlo validation harness.

1) Builds a small random connected frame graph with uncertain SE(3) edges.
2) Adds several points attached to random frames (with intrinsic point covariance).
3) Runs multiple random point-to-point queries into random query frames.
4) For each query, compares:
      - analytic independent baseline
      - analytic correlation-aware (shared edge_id cross-cov)
      - Monte Carlo sample covariance

Notes
-----
- Uses the same modeling assumptions as the library:
  * different edge uncertainties independent
  * shared edges induce correlation for point-to-point deltas via shared edge_id
- For Monte Carlo, inverse edges are handled consistently:
  * each edge_id is sampled once as a forward transform
  * any traversal in the inverse direction uses the exact matrix inverse of that sample
"""

from __future__ import annotations

import numpy as np

from uncertainty_networks import UncertainTransform, GeometricNetwork
from uncertainty_networks.se3 import make_se3, exp_se3


def cov_sample(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / (X.shape[0] - 1)


def rel_frob(A: np.ndarray, B: np.ndarray, eps: float = 1e-18) -> float:
    denom = float(np.linalg.norm(B, ord="fro"))
    if denom < eps:
        return float(np.linalg.norm(A - B, ord="fro"))
    return float(np.linalg.norm(A - B, ord="fro") / denom)


def random_small_rotation(rng: np.random.Generator, sigma: float = 0.08) -> np.ndarray:
    """
    Sample a small rotation using axis-angle ~ N(0, sigma^2 I) and Exp(so(3)).
    We reuse exp_se3 by embedding as [alpha; 0].
    """
    alpha = rng.normal(0.0, sigma, size=(3,))
    xi = np.hstack([alpha, np.zeros(3)])
    T = exp_se3(xi)
    return T[:3, :3]


def build_random_connected_network(
    rng: np.random.Generator,
    n_frames: int = 6,
    extra_edges: int = 4,
) -> tuple[GeometricNetwork, dict]:
    """
    Returns
    -------
    net : GeometricNetwork
    edge_catalog : dict edge_id -> dict with fields:
        - src, dst : canonical forward direction
        - T_nom : nominal forward 4x4
        - C : 6x6
    """
    net = GeometricNetwork()

    frames = [f"F{i}" for i in range(n_frames)]
    for f in frames:
        net.add_frame(f)

    edge_catalog = {}

    # 1) Make a random spanning tree to ensure connectivity
    for i in range(1, n_frames):
        src = frames[rng.integers(0, i)]
        dst = frames[i]

        R = random_small_rotation(rng, sigma=0.06)
        t = rng.normal(0.0, 0.10, size=(3,))  # ~10 cm scale
        T_nom = make_se3(R, t)

        # Small covariances (first-order should match)
        rot_var = rng.uniform(1e-6, 6e-6)
        trans_var = rng.uniform(1e-6, 8e-6)
        C = np.diag([rot_var, rot_var, rot_var, trans_var, trans_var, trans_var])

        U = UncertainTransform(T_nom, C)
        edge_id = net.add_edge(src, dst, U, add_inverse=True)

        edge_catalog[edge_id] = {"src": src, "dst": dst, "T_nom": T_nom, "C": C}

    # 2) Add extra random edges
    all_pairs = [(a, b) for a in frames for b in frames if a != b]
    rng.shuffle(all_pairs)

    count = 0
    for (src, dst) in all_pairs:
        if count >= extra_edges:
            break
        # skip if already present
        if net.get_edge(src, dst) is not None:
            continue

        R = random_small_rotation(rng, sigma=0.06)
        t = rng.normal(0.0, 0.10, size=(3,))
        T_nom = make_se3(R, t)

        rot_var = rng.uniform(1e-6, 6e-6)
        trans_var = rng.uniform(1e-6, 8e-6)
        C = np.diag([rot_var, rot_var, rot_var, trans_var, trans_var, trans_var])

        U = UncertainTransform(T_nom, C)
        edge_id = net.add_edge(src, dst, U, add_inverse=True)
        edge_catalog[edge_id] = {"src": src, "dst": dst, "T_nom": T_nom, "C": C}
        count += 1

    return net, edge_catalog


def build_points(
    rng: np.random.Generator,
    net: GeometricNetwork,
    n_points: int = 4,
) -> list[str]:
    frames = list(net._adj.keys())  # internal but fine for script
    point_names = []

    for i in range(n_points):
        frame = frames[rng.integers(0, len(frames))]

        p_local = rng.normal(0.0, 0.08, size=(3,))  # ~8 cm local scale
        # intrinsic point covariance
        var = rng.uniform(1e-6, 5e-6)
        Cp = var * np.eye(3)

        name = f"p{i}"
        net.add_point(name, frame, p_local, Cp)
        point_names.append(name)

    return point_names


def precompute_path_edge_sequence(
    net: GeometricNetwork,
    edge_catalog: dict,
    start: str,
    goal: str,
) -> list[tuple[str, int]]:
    """
    Returns a list of (edge_id, dir_sign) along a path.
    dir_sign = +1 if traversal matches canonical (src->dst), -1 if inverse.

    Uses net.find_path and net.get_edge_obj (edge_id stored inside).
    """
    path = net.find_path(start, goal)
    if not path:
        raise ValueError(f"No path from {start} to {goal}")

    seq = []
    for a, b in zip(path[:-1], path[1:]):
        e = net.get_edge_obj(a, b)
        if e is None:
            raise RuntimeError("Internal error: missing edge on found path.")
        eid = e.edge_id
        canon = edge_catalog[eid]
        sign = +1 if (a == canon["src"] and b == canon["dst"]) else -1
        seq.append((eid, sign))
    return seq


def apply_edge_sequence(
    seq: list[tuple[str, int]],
    sampled_forward: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Compose a transform matrix along an edge sequence using sampled_forward matrices.

    sampled_forward[eid] gives the sampled transform in canonical forward direction.
    If sign=-1, we use exact inverse of that sampled transform.
    """
    T = np.eye(4)
    for eid, sign in seq:
        T_e = sampled_forward[eid]
        if sign == -1:
            T_e = np.linalg.inv(T_e)
        T = T @ T_e
    return T


def main():
    # Configuration
    seed = 21
    rng = np.random.default_rng(seed)

    n_frames = 6
    extra_edges = 5
    n_points = 5
    n_queries = 18

    N_mc = 25000  # MC samples per query
    verbose_each_query = False

    # Build random network + points
    net, edge_catalog = build_random_connected_network(rng, n_frames=n_frames, extra_edges=extra_edges)
    point_names = build_points(rng, net, n_points=n_points)
    frames = list(net._adj.keys())

    # Build random queries: (src_point, dst_point, query_frame)
    queries = []
    for _ in range(n_queries):
        src, dst = rng.choice(point_names, size=2, replace=False)
        qf = frames[rng.integers(0, len(frames))]
        queries.append((src, dst, qf))

    # Precompute each point's attached frame + local info
    point_meta = {}
    for pn in point_names:
        pnode = net._points[pn]  # internal but fine for script
        point_meta[pn] = {"frame": pnode.frame, "p_local": pnode.p_local, "Cp": pnode.Cp}

    # Precompute path sequences for each (point, query_frame) used
    path_cache: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for (src, dst, qf) in queries:
        for p in (src, dst):
            key = (p, qf)
            if key in path_cache:
                continue
            start_frame = point_meta[p]["frame"]
            path_cache[key] = precompute_path_edge_sequence(net, edge_catalog, start_frame, qf)

    # Run queries
    improvements = 0
    errs_ind = []
    errs_corr = []

    for qi, (src, dst, qf) in enumerate(queries):
        # Analytic
        delta_ind, C_ind = net.query_relative_vector_independent(src, dst, qf)
        delta_corr, C_corr = net.query_relative_vector(src, dst, qf)

        # MC samples of delta
        deltas = np.zeros((N_mc, 3), dtype=float)
        mean0_6 = np.zeros(6)
        mean0_3 = np.zeros(3)

        # Precompute sequences
        seq_src = path_cache[(src, qf)]
        seq_dst = path_cache[(dst, qf)]

        p_src_local = point_meta[src]["p_local"]
        Cp_src = point_meta[src]["Cp"]
        p_dst_local = point_meta[dst]["p_local"]
        Cp_dst = point_meta[dst]["Cp"]

        # MC loop
        for k in range(N_mc):
            # Sample every edge_id once (forward canonical)
            sampled_forward = {}
            for eid, info in edge_catalog.items():
                eta = rng.multivariate_normal(mean0_6, info["C"])
                sampled_forward[eid] = exp_se3(eta) @ info["T_nom"]

            # Transform points to query frame
            T_src_q = apply_edge_sequence(seq_src, sampled_forward)
            R = T_src_q[:3, :3]
            t = T_src_q[:3, 3]
            dp = rng.multivariate_normal(mean0_3, Cp_src)
            p_src_q = R @ (p_src_local + dp) + t

            T_dst_q = apply_edge_sequence(seq_dst, sampled_forward)
            R = T_dst_q[:3, :3]
            t = T_dst_q[:3, 3]
            dp = rng.multivariate_normal(mean0_3, Cp_dst)
            p_dst_q = R @ (p_dst_local + dp) + t

            deltas[k, :] = (p_dst_q - p_src_q)

        C_mc = cov_sample(deltas)

        err_ind = rel_frob(C_ind, C_mc)
        err_corr = rel_frob(C_corr, C_mc)

        errs_ind.append(err_ind)
        errs_corr.append(err_corr)

        if err_corr <= err_ind:
            improvements += 1

        if verbose_each_query:
            print(f"\nQuery {qi+1}/{len(queries)}: (src={src}, dst={dst}, qf={qf})")
            print(f"  rel frob err independent: {err_ind:.4f}")
            print(f"  rel frob err corr-aware : {err_corr:.4f}")
            print(f"  trace(C_ind)={np.trace(C_ind):.3e}  trace(C_corr)={np.trace(C_corr):.3e}  trace(C_mc)={np.trace(C_mc):.3e}")

    errs_ind = np.array(errs_ind)
    errs_corr = np.array(errs_corr)

    print("\n=== Random Network MC Harness Summary ===")
    print(f"Seed: {seed}")
    print(f"Frames: {n_frames}, Extra edges: {extra_edges}")
    print(f"Points: {n_points}, Queries: {n_queries}")
    print(f"MC samples per query: {N_mc}")
    print("")
    print(f"Mean rel frob error (independent): {errs_ind.mean():.4f}")
    print(f"Mean rel frob error (corr-aware) : {errs_corr.mean():.4f}")
    print(f"Worst rel frob error (independent): {errs_ind.max():.4f}")
    print(f"Worst rel frob error (corr-aware) : {errs_corr.max():.4f}")
    print(f"Corr-aware <= independent in {improvements}/{n_queries} queries")


if __name__ == "__main__":
    main()
