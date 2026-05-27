"""
Self-intersection curves within a mesh

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Tuple
from .. import _trueform
from .._spatial import Mesh
from .._core import OffsetBlockedArray
from .._dispatch import InputMeta, build_suffix

_MODE_MAP = {"sos": 1, "primitives": 2}
_RESOLVE_CROSSINGS = 4
_RESOLVE_SELF_CROSSINGS = 8


def self_intersection_curves(
    mesh: Mesh,
    *,
    mode: str = "sos",
    tolerance: float = 0.0,
    resolve_crossings: bool = True,
    resolve_self_crossings: bool = True
) -> Tuple[OffsetBlockedArray, np.ndarray]:
    """
    Find self-intersection curves within a 3D mesh.

    Parameters
    ----------
    mesh : Mesh
        3D mesh to check for self-intersections.
    mode : str, default "sos"
        Intersection mode. "sos" = SoS (fast), "primitives" = handles
        shared edges/vertices.
    tolerance : float, default 0.0
        World-coordinate distance band for predicate tolerance (0 = exact).
    resolve_crossings : bool, default True
        Resolve crossings between different contours on the same face.
    resolve_self_crossings : bool, default True
        Resolve self-crossings within a single contour.

    Returns
    -------
    paths : OffsetBlockedArray
        Paths as indices into the points array. Each path is one curve.
    points : np.ndarray
        Curve point coordinates with shape (N, 3).
    """

    if not isinstance(mesh, Mesh):
        raise TypeError(
            f"mesh must be a Mesh object, got {type(mesh).__name__}. "
            f"Topology information is required for self-intersection curves."
        )

    if mesh.dims != 3:
        raise ValueError(
            f"self_intersection_curves only supports 3D meshes, got mesh with {mesh.dims}D"
        )

    if mode not in _MODE_MAP:
        raise ValueError(f"mode must be 'sos' or 'primitives', got '{mode}'")

    m = _MODE_MAP[mode]
    if resolve_crossings:
        m |= _RESOLVE_CROSSINGS
    if resolve_self_crossings:
        m |= _RESOLVE_SELF_CROSSINGS

    ngon = 'dyn' if mesh.is_dynamic else str(mesh.ngon)
    meta = InputMeta(mesh.faces.dtype, mesh.dtype, ngon, 3)
    suffix = build_suffix(meta)

    func_name = f"self_intersection_curves_mesh_{suffix}"
    (paths_offsets, paths_data), points = getattr(_trueform.intersect, func_name)(
        mesh._wrapper, m, tolerance
    )

    return OffsetBlockedArray(paths_offsets, paths_data), points
