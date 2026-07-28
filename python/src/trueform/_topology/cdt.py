"""
Constrained Delaunay triangulation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import Optional, Tuple, Union
import numpy as np
from .. import _trueform


def cdt(
    points: np.ndarray,
    edges: Optional[np.ndarray] = None,
    *,
    edge_mask: Optional[np.ndarray] = None,
    split_constraints: bool = True,
    return_index_map: bool = False,
):
    """
    Constrained Delaunay triangulation of 2D points.

    Without `edges`, returns the convex-hull Delaunay triangulation.
    With `edges`, recovers each constraint as an edge of the output;
    intersecting constraints (VV / VE / EE) are split automatically and
    intersection vertices are added to the output points.

    Pass `split_constraints=False` to demand the constraints verbatim.
    The triangulation then refuses to create a point where two of them
    cross and returns an empty result — that emptiness is the answer to
    "do these constraints cross", not an error.

    Parameters
    ----------
    points : np.ndarray, shape (N, 2), dtype float32 or float64
        Input 2D points.
    edges : np.ndarray, shape (M, 2), dtype int32, optional
        Constraint edges as pairs of indices into `points`. When omitted,
        the result is the unconstrained convex-hull Delaunay triangulation.
    edge_mask : np.ndarray, shape (M,), dtype bool, optional, keyword-only
        Per-edge boundary flag. `True` marks the constraint as a region
        wall (parity flips when crossed during interior extraction);
        `False` marks it as preserved-but-not-boundary. Defaults to all
        `True` when omitted.
    split_constraints : bool, optional, keyword-only
        `True` (default) resolves a crossing between two constraints by
        adding an output point there. `False` keeps every constraint
        whole and returns empty `faces` and `points` if any two cross.
        Requires `edges`.
    return_index_map : bool, optional, keyword-only
        If `True`, also returns the input-to-output point index map.

    Returns
    -------
    Without `return_index_map`:
        (faces, points) : tuple
            faces  : (K, 3) int32 array of triangle indices into points
            points : (P, 2) array of output point coordinates (dtype matches
                     input)

    With `return_index_map`:
        ((faces, points), (f, kept_ids)) : tuple
            f         : (N,) int32 array — output index of each input point
            kept_ids  : (P,) int32 array — input index that survived for
                        each output slot, or `f.size()` for synthetic
                        intersection vertices.
    """
    if not isinstance(points, np.ndarray):
        raise TypeError(f"points must be np.ndarray, got {type(points).__name__}")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points.shape}")
    if points.dtype == np.float32:
        suffix = "float"
    elif points.dtype == np.float64:
        suffix = "double"
    else:
        raise TypeError(
            f"points dtype must be float32 or float64, got {points.dtype}"
        )

    if edges is None:
        if edge_mask is not None:
            raise ValueError("edge_mask requires edges to be provided")
        if not split_constraints:
            raise ValueError("split_constraints requires edges to be provided")
        if return_index_map:
            return getattr(_trueform.topology, f"make_cdt_{suffix}_with_maps")(points)
        return getattr(_trueform.topology, f"make_cdt_{suffix}")(points)

    if not isinstance(edges, np.ndarray):
        raise TypeError(f"edges must be np.ndarray, got {type(edges).__name__}")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must have shape (M, 2), got {edges.shape}")
    if edges.dtype != np.int32:
        edges = np.ascontiguousarray(edges, dtype=np.int32)

    if edge_mask is not None:
        if not isinstance(edge_mask, np.ndarray):
            raise TypeError(
                f"edge_mask must be np.ndarray, got {type(edge_mask).__name__}"
            )
        if edge_mask.ndim != 1 or edge_mask.shape[0] != edges.shape[0]:
            raise ValueError(
                f"edge_mask must have shape ({edges.shape[0]},), "
                f"got {edge_mask.shape}"
            )
        if edge_mask.dtype != np.bool_:
            edge_mask = np.ascontiguousarray(edge_mask, dtype=np.bool_)

    if return_index_map:
        func = getattr(_trueform.topology, f"make_cdt_{suffix}_edges_with_maps")
    else:
        func = getattr(_trueform.topology, f"make_cdt_{suffix}_edges")
    return func(points, edges, edge_mask, split_constraints)
