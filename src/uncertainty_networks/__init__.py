"""
Author: X.M. Christine Zhu
Date: 04/04/2026

Unified Gaussian linear system framework for uncertainty propagation
through geometric relationships on SE(3), following the CIS I
left-multiplicative perturbation convention and the structural
identification approach from Doc 2.

The main entry points are:

    GeometricNetwork   — build a frame graph, add edges and points
    query_frame()      — fused multi-path query (handles all topologies)
    query()            — single shortest-path query
    query_point()      — propagate a point into any frame
    query_relative_vector() — correlation-aware relative vector
    query_distance()   — Euclidean distance with first-order variance

Mathematical reference: docs/math_note.pdf
"""

from .uncertain_geometry import UncertainTransform
from .network import GeometricNetwork, PathResult, FusedQueryResult

__all__ = [
    "UncertainTransform",
    "GeometricNetwork",
    "PathResult",
    "FusedQueryResult",
]