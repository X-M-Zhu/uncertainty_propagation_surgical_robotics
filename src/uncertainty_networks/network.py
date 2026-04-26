# Author: X.M. Christine Zhu
# Date: 04/04/2026

"""
GeometricNetwork: unified Gaussian linear system for uncertain geometric frames.

Mathematical framework (Doc 2 — Structural Identification)
----------------------------------------------------------
Each unique physical edge e has ONE latent variable

    η_{e,canonical} ~ N(0, C_e)

shared by every path that traverses it.  This is the structural identity index
α(i, e) from Doc 2: two paths with the same edge_id at the same position share
the same draw, not independent copies.

For m simple paths from start to goal, each path i gives one linear estimate
of the unknown transform perturbation η_0:

    b_i = Σ_e  A_e^i  η_{e,canonical}

where A_e^i maps the canonical edge variable to the path perturbation.

The full stacked covariance S (6m × 6m) follows Doc 2 Section 1:

    S_{iℓ} = Σ_e  A_e^i  C_e  (A_e^ℓ)^T  · 1{α(i,e) = α(ℓ,e)}

The indicator is 1 when paths i and ℓ share the same physical edge (same
edge_id); the off-diagonal blocks S_{iℓ} are generally non-zero.

With A_0 = [I; I; ...; I] (all paths estimate the same η_0), the information
form (Doc 2 Section 2.2) gives:

    C_0 = (A_0^T S^{-1} A_0)^{-1} = (Σ_{i,ℓ} [S^{-1}]_{iℓ})^{-1}

This single formula handles every topology automatically:
  - Open chain      (1 path)  → standard covariance propagation
  - Independent multipaths    → reduces to information-form fusion
  - Multipaths with shared edges → off-diagonal S_{iℓ} prevent double-counting
  - Closed loops              → same as multipath, no special treatment needed

Point queries
-------------
Cross-covariance between two points in the same frame is computed via
shared-edge Jacobians (same structural identification: same edge_id → same
physical uncertainty contributes to both points).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from .uncertain_geometry import UncertainTransform
from .se3 import adjoint_se3, inv_se3, skew


Array = np.ndarray


def _sym(A: Array) -> Array:
    """Symmetrise a square matrix."""
    return 0.5 * (A + A.T)


# ------------------------------------------------------------------ #
#  Data containers                                                     #
# ------------------------------------------------------------------ #

@dataclass
class PathResult:
    """
    Result of a single-path frame-to-frame query.

    Attributes
    ----------
    path : list[str]
        Frame names [start, ..., goal].
    transform : UncertainTransform
        Composed nominal transform and propagated covariance along this path.
    edge_ids : list[str]
        edge_id for each hop (length = len(path) - 1).
    certain_mask : list[bool]
        is_certain flag per hop.
    forward_mask : list[bool]
        is_forward flag per hop (True = canonical direction).
    edge_types : list[str]
        edge_type per hop: "se3", "rot_only", "trans_only", or "vector".
    """
    path: List[str]
    transform: UncertainTransform
    edge_ids: List[str] = field(default_factory=list)
    certain_mask: List[bool] = field(default_factory=list)
    forward_mask: List[bool] = field(default_factory=list)
    edge_types: List[str] = field(default_factory=list)


@dataclass
class FusedQueryResult:
    """
    Result of the unified multi-path query (query_frame).

    The covariance C_0 is computed from the full stacked covariance S using
    the information form from Doc 2.  Off-diagonal blocks of S correctly
    account for shared physical edges between paths.

    Attributes
    ----------
    transform : UncertainTransform
        Best-estimate transform.  Nominal from first path (all paths agree
        nominally in a consistent network).  C is the unified C_0.
    n_paths : int
        Number of simple paths found and used.
    path_results : list[PathResult]
        Per-path results before fusion (for diagnostics).
    """
    transform: UncertainTransform
    n_paths: int
    path_results: List[PathResult]


@dataclass
class PointNode:
    """A 3-D point rigidly attached to a coordinate frame."""
    frame: str
    p_local: Array   # (3,) nominal local coordinates
    Cp: Array        # (3,3) local covariance


@dataclass
class Edge:
    """
    Directed edge storing an uncertain transform.

    edge_id : str
        Unique identifier shared by the forward and inverse edge.
        This IS the structural identity index α(i, j) from Doc 2:
        all paths that use this physical edge share one latent variable.
    is_certain : bool
        True if this edge is perfectly known (C ≈ 0).
    is_forward : bool
        True if this is the canonical direction the edge was added.
        The inverse edge has the same edge_id with is_forward=False.
    edge_type : str
        "se3", "rot_only", "trans_only", or "vector".
    """
    edge_id: str
    transform: UncertainTransform
    is_certain: bool = False
    is_forward: bool = True
    edge_type: str = "se3"


_VALID_EDGE_TYPES = frozenset({"se3", "rot_only", "trans_only", "vector"})


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _project_covariance(C: Array, edge_type: str) -> Array:
    """Zero out covariance blocks that are inactive for the given edge type."""
    C = np.array(C, dtype=float)
    if edge_type == "rot_only":
        C[3:, :] = 0.0
        C[:, 3:] = 0.0
    elif edge_type in ("trans_only", "vector"):
        C[:3, :] = 0.0
        C[:, :3] = 0.0
    return C


# ------------------------------------------------------------------ #
#  GeometricNetwork                                                    #
# ------------------------------------------------------------------ #

class GeometricNetwork:
    """
    Directed graph of coordinate frames connected by uncertain SE(3) transforms.

    Implements the unified Gaussian linear system framework from Doc 2.
    The main query entry point is query_frame(), which finds all simple paths
    and returns the correct fused covariance for any network topology.
    """

    def __init__(self) -> None:
        self._adj: Dict[str, Dict[str, Edge]] = {}
        self._points: Dict[str, PointNode] = {}
        self._edge_counter: int = 0

    # ---------------------------------------------------------------- #
    #  Frame / edge construction                                        #
    # ---------------------------------------------------------------- #

    def add_frame(self, name: str) -> None:
        if name not in self._adj:
            self._adj[name] = {}

    def add_edge(
        self,
        src: str,
        dst: str,
        T_src_dst: UncertainTransform,
        add_inverse: bool = True,
        is_certain: bool = False,
        edge_type: str = "se3",
    ) -> str:
        """
        Add a directed edge src -> dst.

        The forward and inverse edges share the same edge_id, which is the
        structural identity index α from Doc 2: all paths that traverse this
        physical relationship share the same latent variable η_{e,canonical}.

        Parameters
        ----------
        src, dst : str
        T_src_dst : UncertainTransform
        add_inverse : bool
            Also add the reverse edge (same edge_id, is_forward=False).
        is_certain : bool
            Mark edge as perfectly known.
        edge_type : {"se3", "rot_only", "trans_only", "vector"}

        Returns
        -------
        edge_id : str
        """
        if edge_type not in _VALID_EDGE_TYPES:
            raise ValueError(
                f"edge_type must be one of {sorted(_VALID_EDGE_TYPES)}, got '{edge_type}'"
            )

        self.add_frame(src)
        self.add_frame(dst)

        edge_id = f"e{self._edge_counter:06d}"
        self._edge_counter += 1

        C_proj = _project_covariance(T_src_dst.C, edge_type)

        F_nom = T_src_dst.F_nom
        if edge_type == "vector":
            F_nom = np.array(F_nom, dtype=float)
            F_nom[:3, :3] = np.eye(3)

        T_fwd = UncertainTransform(F_nom, C_proj)

        self._adj[src][dst] = Edge(
            edge_id=edge_id,
            transform=T_fwd,
            is_certain=is_certain,
            is_forward=True,
            edge_type=edge_type,
        )

        if add_inverse:
            self._adj[dst][src] = Edge(
                edge_id=edge_id,
                transform=T_fwd.inv(),
                is_certain=is_certain,
                is_forward=False,
                edge_type=edge_type,
            )

        return edge_id

    # ---------------------------------------------------------------- #
    #  Point management                                                 #
    # ---------------------------------------------------------------- #

    def add_point(
        self,
        name: str,
        frame: str,
        p_local: Array,
        Cp: Array,
    ) -> None:
        if frame not in self._adj:
            raise KeyError(f"Unknown frame '{frame}'")
        self._points[name] = PointNode(
            frame=frame,
            p_local=np.asarray(p_local, dtype=float).reshape(3),
            Cp=np.asarray(Cp, dtype=float).reshape(3, 3),
        )

    def has_point(self, name: str) -> bool:
        return name in self._points

    # ---------------------------------------------------------------- #
    #  Basic utilities                                                  #
    # ---------------------------------------------------------------- #

    def has_frame(self, name: str) -> bool:
        return name in self._adj

    def frames(self) -> List[str]:
        """All frame names in insertion order."""
        return list(self._adj.keys())

    def forward_edges(self) -> List[Tuple[str, str]]:
        """All edges in their canonical (forward) direction as (src, dst) pairs."""
        result = []
        for src, neighbors in self._adj.items():
            for dst, edge in neighbors.items():
                if edge.is_forward:
                    result.append((src, dst))
        return result

    def neighbors(self, name: str) -> List[str]:
        return list(self._adj.get(name, {}).keys())

    def get_edge(self, src: str, dst: str) -> Optional[UncertainTransform]:
        e = self._adj.get(src, {}).get(dst, None)
        return None if e is None else e.transform

    def get_edge_obj(self, src: str, dst: str) -> Optional[Edge]:
        return self._adj.get(src, {}).get(dst, None)

    # ---------------------------------------------------------------- #
    #  Path finding                                                     #
    # ---------------------------------------------------------------- #

    def find_path(self, start: str, goal: str) -> List[str]:
        """BFS shortest path from start to goal."""
        if start not in self._adj or goal not in self._adj:
            return []
        if start == goal:
            return [start]

        q = deque([start])
        parent: Dict[str, Optional[str]] = {start: None}

        while q:
            cur = q.popleft()
            for nxt in self._adj[cur]:
                if nxt in parent:
                    continue
                parent[nxt] = cur
                if nxt == goal:
                    q.clear()
                    break
                q.append(nxt)

        if goal not in parent:
            return []

        path: List[str] = []
        node: Optional[str] = goal
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path

    def _edges_along_path(self, path: List[str]) -> List[Edge]:
        return [self._adj[a][b] for a, b in zip(path[:-1], path[1:])]

    def find_all_paths(
        self,
        start: str,
        goal: str,
        max_depth: Optional[int] = None,
    ) -> List[List[str]]:
        """
        Find all simple paths from start to goal via DFS.
        A simple path visits each frame at most once (no cycles).
        """
        if start not in self._adj or goal not in self._adj:
            return []
        if start == goal:
            return [[start]]

        all_paths: List[List[str]] = []

        def _dfs(current: str, path: List[str], visited: set) -> None:
            if current == goal:
                all_paths.append(list(path))
                return
            if max_depth is not None and len(path) > max_depth:
                return
            for neighbor in self._adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    _dfs(neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)

        _dfs(start, [start], {start})
        return all_paths

    def query_all_paths(
        self,
        start: str,
        goal: str,
        max_depth: Optional[int] = None,
    ) -> List[PathResult]:
        """
        Compute a PathResult for every simple path from start to goal.

        The per-path covariance stored in each PathResult is the standard
        composed covariance (diagonal term S_{ii} from the S matrix).
        For the fully fused result use query_frame().
        """
        if start not in self._adj:
            raise KeyError(f"Unknown start frame: {start}")
        if goal not in self._adj:
            raise KeyError(f"Unknown goal frame: {goal}")

        paths = self.find_all_paths(start, goal, max_depth=max_depth)
        results: List[PathResult] = []

        for path in paths:
            edges = self._edges_along_path(path)
            T = UncertainTransform.identity()
            for e in edges:
                T = T @ e.transform
            results.append(PathResult(
                path=path,
                transform=T,
                edge_ids=[e.edge_id for e in edges],
                certain_mask=[e.is_certain for e in edges],
                forward_mask=[e.is_forward for e in edges],
                edge_types=[e.edge_type for e in edges],
            ))

        return results

    # ---------------------------------------------------------------- #
    #  Single-path frame queries                                        #
    # ---------------------------------------------------------------- #

    def query(self, start: str, goal: str) -> PathResult:
        """
        Query via the BFS shortest path only (single path, no multi-path fusion).
        For the correct fused result over all paths use query_frame().
        """
        if start not in self._adj:
            raise KeyError(f"Unknown start frame: {start}")
        if goal not in self._adj:
            raise KeyError(f"Unknown goal frame: {goal}")

        path = self.find_path(start, goal)
        if not path:
            raise ValueError(f"No path found from '{start}' to '{goal}'")

        edges = self._edges_along_path(path)
        T = UncertainTransform.identity()
        for e in edges:
            T = T @ e.transform

        return PathResult(
            path=path,
            transform=T,
            edge_ids=[e.edge_id for e in edges],
            certain_mask=[e.is_certain for e in edges],
            forward_mask=[e.is_forward for e in edges],
            edge_types=[e.edge_type for e in edges],
        )

    def query_transform_on_path(self, path: List[str]) -> PathResult:
        """Compute the uncertain transform along a user-specified path."""
        if not path:
            raise ValueError("Path is empty.")
        for v in path:
            if v not in self._adj:
                raise KeyError(f"Unknown frame in path: {v}")
        for a, b in zip(path[:-1], path[1:]):
            if b not in self._adj.get(a, {}):
                raise ValueError(f"No edge {a} -> {b} in network.")

        edges = self._edges_along_path(path)
        T = UncertainTransform.identity()
        for e in edges:
            T = T @ e.transform

        return PathResult(
            path=path,
            transform=T,
            edge_ids=[e.edge_id for e in edges],
            certain_mask=[e.is_certain for e in edges],
            forward_mask=[e.is_forward for e in edges],
            edge_types=[e.edge_type for e in edges],
        )

    # ---------------------------------------------------------------- #
    #  Unified multi-path query — core of the framework                 #
    # ---------------------------------------------------------------- #

    def _compute_path_canonical_terms(
        self,
        path: List[str],
    ) -> Dict[str, Tuple[Array, Array]]:
        """
        For each edge in the path, compute the canonical linear map and
        canonical covariance, following Doc 2 (structural identification).

        The canonical variable η_{e,canonical} is always defined in the
        FORWARD direction of the edge.  For a path that traverses the edge
        in the inverse direction the effective linear map picks up an extra
        factor so that the cross-covariance formula remains correct.

        Returns
        -------
        dict  edge_id -> (A_canonical, C_canonical)

        A_canonical : (6,6)
            Linear map:  η_0 contribution from this edge = A_canonical @ η_{e,canonical}

            Forward traversal (is_forward=True):
                A_canonical = Ad_{T_prefix}

            Inverse traversal (is_forward=False):
                Under CIS I, inverting a transform gives
                    η_inv ≈ -Ad_{F_inv} η_fwd    (first-order)
                so:
                    A_canonical = -Ad_{T_prefix} @ Ad_{F_inv_nom}

        C_canonical : (6,6)
            Covariance of the canonical (forward) variable.
            For forward edges this is just C_e.
            For inverse edges it is recovered as:
                C_canonical = Ad_{F_fwd} C_inv Ad_{F_fwd}^T
            where F_fwd = inv(F_inv_nom).
        """
        edges = self._edges_along_path(path)
        T_prefix = np.eye(4, dtype=float)
        result: Dict[str, Tuple[Array, Array]] = {}

        for e in edges:
            Ad_prefix = adjoint_se3(T_prefix)

            if e.is_forward:
                # Canonical variable is η_fwd, contribution = Ad_prefix @ η_fwd
                A_can = Ad_prefix.copy()
                C_can = e.transform.C.copy()
            else:
                # η_inv ≈ -Ad_{F_inv} η_fwd  →  contribution = Ad_prefix @ η_inv
                #                                             = -Ad_prefix @ Ad_{F_inv} @ η_fwd
                Ad_Finv = adjoint_se3(e.transform.F_nom)
                A_can = -Ad_prefix @ Ad_Finv

                # Recover C_canonical from stored C_inv:
                #   C_inv = Ad_{F_inv} C_fwd Ad_{F_inv}^T
                #   C_fwd = Ad_{F_fwd} C_inv Ad_{F_fwd}^T,   F_fwd = inv(F_inv_nom)
                Ad_Ffwd = adjoint_se3(inv_se3(e.transform.F_nom))
                C_can = _sym(Ad_Ffwd @ e.transform.C @ Ad_Ffwd.T)

            result[e.edge_id] = (A_can, C_can)
            T_prefix = T_prefix @ e.transform.F_nom

        return result

    def query_frame(
        self,
        start: str,
        goal: str,
        max_depth: Optional[int] = None,
    ) -> FusedQueryResult:
        """
        Query the best-estimate transform from start to goal using ALL simple
        paths and the unified Gaussian linear system framework (Doc 2).

        Algorithm
        ---------
        1. Find all m simple paths from start to goal.

        2. For each path i compute canonical terms (A_e^i, C_e) per edge.

        3. Build the full stacked covariance S  (6m × 6m):

               S_{iℓ} = Σ_{shared edges e}  A_e^i  C_e  (A_e^ℓ)^T

           "Shared" means same physical edge_id — the structural identity
           α(i, e) = α(ℓ, e) from Doc 2.

        4. Apply the information form (Doc 2 Section 2.2):

               C_0 = (A_0^T S^{-1} A_0)^{-1}

           With A_0 = [I; I; ...; I]:

               C_0 = ( Σ_{i,ℓ} [S^{-1}]_{iℓ} )^{-1}

        Special cases handled automatically
        ------------------------------------
        m = 1  (open chain)          : C_0 = S_{11} = path covariance.
        m > 1, no shared edges       : S block-diagonal → standard info fusion.
        m > 1, shared edges present  : off-diagonal S_{iℓ} prevent double-
                                       counting shared uncertainty.
        Closed loops                 : same as multipath — no special treatment.

        Parameters
        ----------
        start, goal : str
        max_depth : int, optional
            Limit path search depth (useful for large or dense networks).

        Returns
        -------
        FusedQueryResult
        """
        if start not in self._adj:
            raise KeyError(f"Unknown start frame: '{start}'")
        if goal not in self._adj:
            raise KeyError(f"Unknown goal frame: '{goal}'")

        # Step 1 — find all simple paths and per-path results.
        path_results = self.query_all_paths(start, goal, max_depth=max_depth)
        if not path_results:
            raise ValueError(f"No path found from '{start}' to '{goal}'")

        m = len(path_results)

        # Step 2 — single path: trivially correct, no matrix inversion needed.
        if m == 1:
            return FusedQueryResult(
                transform=path_results[0].transform,
                n_paths=1,
                path_results=path_results,
            )

        # Step 3 — canonical terms per path:  edge_id -> (A_e^i, C_canonical_e).
        canonical_terms = [
            self._compute_path_canonical_terms(pr.path)
            for pr in path_results
        ]

        # Step 4 — build full S matrix (6m × 6m).
        #
        # S_{iℓ} = Σ_{e: α(i,e)=α(ℓ,e)}  A_e^i @ C_canonical_e @ (A_e^ℓ)^T
        #
        # For the diagonal (i == ℓ) this equals the per-path composed covariance.
        # For off-diagonal blocks it captures shared-edge correlations.
        S = np.zeros((6 * m, 6 * m), dtype=float)

        for i in range(m):
            for l in range(m):
                S_il = np.zeros((6, 6), dtype=float)
                shared_eids = (
                    set(canonical_terms[i].keys()) & set(canonical_terms[l].keys())
                )
                for eid in shared_eids:
                    A_i, C_can = canonical_terms[i][eid]
                    A_l, _     = canonical_terms[l][eid]
                    S_il += A_i @ C_can @ A_l.T
                S[6 * i : 6 * i + 6, 6 * l : 6 * l + 6] = S_il

        S = _sym(S)

        # Step 5 — information form: C_0 = (A_0^T S^{-1} A_0)^{-1}.
        # A_0 = [I;I;...;I]  →  A_0^T S^{-1} A_0 = sum of all (6×6) blocks of S^{-1}.
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        info = np.zeros((6, 6), dtype=float)
        for i in range(m):
            for l in range(m):
                info += S_inv[6 * i : 6 * i + 6, 6 * l : 6 * l + 6]

        info = _sym(info)

        try:
            C_0 = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            C_0 = np.linalg.pinv(info)

        C_0 = _sym(C_0)

        # Step 6 — nominal: all paths agree in a nominally consistent network.
        F_nom = path_results[0].transform.F_nom

        return FusedQueryResult(
            transform=UncertainTransform(F_nom, C_0),
            n_paths=m,
            path_results=path_results,
        )

    # ---------------------------------------------------------------- #
    #  Point propagation (with per-edge Jacobians)                      #
    # ---------------------------------------------------------------- #

    def _query_point_with_edge_jacobians(
        self,
        point_name: str,
        target_frame: str,
    ) -> Tuple[Array, Array, Dict[str, Tuple[Array, Array]]]:
        """
        Propagate a point into target_frame, returning per-edge Jacobians.

        Returns
        -------
        p_out : (3,)
        Cp_out : (3,3)
        edge_terms : dict  edge_id -> (J_e, C_e)
            J_e : (3,6)  Jacobian of point position w.r.t. edge perturbation η_e
            C_e : (6,6)  covariance of that edge (as stored)
        """
        if point_name not in self._points:
            raise KeyError(f"Unknown point '{point_name}'")
        if target_frame not in self._adj:
            raise KeyError(f"Unknown target frame '{target_frame}'")

        point = self._points[point_name]
        source_frame = point.frame

        if source_frame == target_frame:
            return point.p_local.copy(), point.Cp.copy(), {}

        path = self.find_path(source_frame, target_frame)
        if not path:
            raise ValueError(
                f"No path from '{source_frame}' to '{target_frame}'"
            )

        edges = self._edges_along_path(path)

        # Accumulate prefix transform and collect prefix adjoints per edge.
        T_prefix = np.eye(4, dtype=float)
        Ad_prefix_list: List[Array] = []
        for e in edges:
            Ad_prefix_list.append(adjoint_se3(T_prefix))
            T_prefix = T_prefix @ e.transform.F_nom

        T_total = T_prefix
        R_total = T_total[:3, :3]
        t_total = T_total[:3, 3]

        p_nom = R_total @ point.p_local + t_total

        # CIS I Jacobian of p_nom w.r.t. aggregated pose perturbation η_total.
        J_point = np.zeros((3, 6), dtype=float)
        J_point[:, :3] = -skew(p_nom)          # ∂p/∂α
        J_point[:, 3:] = np.eye(3, dtype=float) # ∂p/∂ε

        # Per-edge Jacobians: J_e = J_point @ Ad_{T_prefix up to e}.
        edge_terms: Dict[str, Tuple[Array, Array]] = {}
        Cp_edges = np.zeros((3, 3), dtype=float)

        for e, Ad_pref in zip(edges, Ad_prefix_list):
            J_e = J_point @ Ad_pref
            C_e = e.transform.C
            Cp_edges += J_e @ C_e @ J_e.T
            edge_terms[e.edge_id] = (J_e, C_e)

        # Intrinsic point covariance mapped by R_total.
        Cp_point = R_total @ point.Cp @ R_total.T
        Cp_out = _sym(Cp_edges + Cp_point)

        return p_nom, Cp_out, edge_terms

    # ---------------------------------------------------------------- #
    #  Point queries                                                    #
    # ---------------------------------------------------------------- #

    def query_point(
        self,
        point_name: str,
        target_frame: str,
    ) -> Tuple[Array, Array]:
        """Query a point into target_frame.  Returns (p_nom, Cp)."""
        p, Cp, _ = self._query_point_with_edge_jacobians(point_name, target_frame)
        return p, Cp

    def query_point_to_point(
        self,
        src_point: str,
        dst_point: str,
        query_frame: str,
    ) -> Tuple[Array, Array, Array, Array]:
        """
        Query two points into a common frame.
        Returns (p_src, Cp_src, p_dst, Cp_dst).
        """
        if query_frame not in self._adj:
            raise KeyError(f"Unknown query frame '{query_frame}'")
        p_src, Cp_src = self.query_point(src_point, query_frame)
        p_dst, Cp_dst = self.query_point(dst_point, query_frame)
        return p_src, Cp_src, p_dst, Cp_dst

    def query_relative_vector_independent(
        self,
        src_point: str,
        dst_point: str,
        query_frame: str,
    ) -> Tuple[Array, Array]:
        """
        Relative vector with the independence approximation (upper bound):
            delta   = p_dst - p_src
            C_delta = C_dst + C_src
        """
        p_src, Cp_src, p_dst, Cp_dst = self.query_point_to_point(
            src_point, dst_point, query_frame
        )
        delta = p_dst - p_src
        return delta, _sym(Cp_dst + Cp_src)

    def query_relative_vector(
        self,
        src_point: str,
        dst_point: str,
        query_frame: str,
    ) -> Tuple[Array, Array]:
        """
        Correlation-aware relative vector (correct).

        Applies the structural identification model: edges with the same
        edge_id contribute a cross-covariance between the two point estimates.

            Cross   = Cov(p_dst, p_src) = Σ_{e shared} J_dst,e @ C_e @ J_src,e^T

            C_delta = C_dst + C_src - Cross - Cross^T

        Parameters
        ----------
        src_point, dst_point : str
            Point names registered with add_point().
        query_frame : str
            The frame in which the relative vector is expressed.

        Returns
        -------
        delta  : (3,)
        C_delta : (3,3)
        """
        if query_frame not in self._adj:
            raise KeyError(f"Unknown query frame '{query_frame}'")

        p_src, Cp_src, terms_src = self._query_point_with_edge_jacobians(
            src_point, query_frame
        )
        p_dst, Cp_dst, terms_dst = self._query_point_with_edge_jacobians(
            dst_point, query_frame
        )

        delta = p_dst - p_src

        # Cross-covariance from shared physical edges (same edge_id).
        cross = np.zeros((3, 3), dtype=float)
        shared_eids = set(terms_src.keys()) & set(terms_dst.keys())
        for eid in shared_eids:
            J_src, C_e = terms_src[eid]
            J_dst, _   = terms_dst[eid]
            cross += J_dst @ C_e @ J_src.T   # Cov(p_dst, p_src)

        C_delta = _sym(Cp_dst + Cp_src - cross - cross.T)
        return delta, C_delta

    def query_distance(
        self,
        src_point: str,
        dst_point: str,
        query_frame: str,
        eps: float = 1e-12,
    ) -> Tuple[float, float]:
        """
        Euclidean distance and its variance using the correlation-aware C_delta.

        Returns
        -------
        d     : float   nominal distance
        var_d : float   first-order variance
        """
        delta, C_delta = self.query_relative_vector(src_point, dst_point, query_frame)
        d = float(np.linalg.norm(delta))

        if d < eps:
            return d, float(np.trace(C_delta))

        J = delta.reshape(1, 3) / d            # (1,3) unit-vector Jacobian
        var_d = float((J @ C_delta @ J.T).item())
        return d, var_d

    # ---------------------------------------------------------------- #
    #  Batch utilities                                                  #
    # ---------------------------------------------------------------- #

    def evaluate_pairs(
        self,
        pairs: List[Tuple[str, str, str]],
        compute_distance: bool = False,
    ) -> dict:
        """
        Evaluate multiple point-to-point queries in one call.

        Parameters
        ----------
        pairs : list of (src_point, dst_point, query_frame)
        compute_distance : bool

        Returns
        -------
        dict  (src, dst, query_frame) -> {
            "delta_ind" : (3,),
            "C_ind"     : (3,3),
            "delta_corr": (3,),
            "C_corr"    : (3,3),
            "d"         : float  (only if compute_distance=True),
            "var_d"     : float  (only if compute_distance=True),
        }
        """
        results = {}
        for (src, dst, q) in pairs:
            delta_ind,  C_ind  = self.query_relative_vector_independent(src, dst, q)
            delta_corr, C_corr = self.query_relative_vector(src, dst, q)
            entry = {
                "delta_ind":  delta_ind,
                "C_ind":      C_ind,
                "delta_corr": delta_corr,
                "C_corr":     C_corr,
            }
            if compute_distance:
                d, var_d = self.query_distance(src, dst, q)
                entry["d"]     = d
                entry["var_d"] = var_d
            results[(src, dst, q)] = entry
        return results
