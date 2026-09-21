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


def self_intersection_curves(
    mesh: Mesh,
    *,
    mode: str = "primitives",
    tolerance: float = 0.0
) -> Tuple[OffsetBlockedArray, np.ndarray]:
    """
    Find self-intersection curves within a 3D mesh.

    Parameters
    ----------
    mesh : Mesh
        3D mesh to check for self-intersections.
    mode : str, default "primitives"
        The classifier the run states its contacts with. "primitives"
        classifies shared edges/vertices and coplanar contacts; "sos"
        perturbs every contact into a crossing and cannot state shared or
        coplanar geometry. A one-mesh build asks for the mesh's own
        self-intersections, so its crossing contours are resolved too.
    tolerance : float, default 0.0
        World-coordinate distance an input vertex may move to reach the lattice
        (0 = exact).

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

    ngon = 'dyn' if mesh.is_dynamic else str(mesh.ngon)
    meta = InputMeta(mesh.faces.dtype, mesh.dtype, ngon, 3)
    suffix = build_suffix(meta)

    func_name = f"self_intersection_curves_mesh_{suffix}"
    (paths_offsets, paths_data), points = getattr(_trueform.intersect, func_name)(
        mesh._wrapper, m, tolerance
    )

    return OffsetBlockedArray(paths_offsets, paths_data), points
