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


def self_intersection_curves(
    mesh: Mesh
) -> Tuple[OffsetBlockedArray, np.ndarray]:
    """
    Find self-intersection curves within a 3D mesh.

    Detects where a mesh intersects itself and returns the intersection curves.

    Parameters
    ----------
    mesh : Mesh
        3D mesh to check for self-intersections

    Returns
    -------
    paths : OffsetBlockedArray
        Paths as indices into the points array. Each path is one curve.
        Iterate over paths to get individual curves: `for path_ids in paths: ...`
    points : np.ndarray
        Curve point coordinates with shape (N, 3)
        Access curve points via: `curve_points = points[paths[i]]`

    Raises
    ------
    ValueError
        If mesh is not 3D
    TypeError
        If input is not a Mesh object

    Examples
    --------
    >>> import trueform as tf
    >>> # Load a mesh that may have self-intersections
    >>> faces, points = tf.read_stl("mesh.stl")
    >>> mesh = tf.Mesh(faces, points)
    >>>
    >>> # Find self-intersection curves
    >>> paths, curve_points = tf.self_intersection_curves(mesh)
    >>> print(f"Found {len(paths)} self-intersection curve(s)")
    >>>
    >>> if len(paths) > 0:
    ...     # Iterate over curves
    ...     for path_ids in paths:
    ...         pts = curve_points[path_ids]
    ...         # Process curve (e.g., visualize, repair, etc.)
    """

    # 1. VALIDATE INPUT IS MESH OBJECT
    if not isinstance(mesh, Mesh):
        raise TypeError(
            f"mesh must be a Mesh object, got {type(mesh).__name__}. "
            f"Topology information is required for self-intersection curves."
        )

    # 2. VALIDATE 3D
    if mesh.dims != 3:
        raise ValueError(
            f"self_intersection_curves only supports 3D meshes, got mesh with {mesh.dims}D"
        )

    # 3. BUILD SUFFIX FOR C++ FUNCTION
    ngon = 'dyn' if mesh.is_dynamic else str(mesh.ngon)
    meta = InputMeta(mesh.faces.dtype, mesh.dtype, ngon, 3)
    suffix = build_suffix(meta)

    # 4. DISPATCH TO C++
    func_name = f"self_intersection_curves_mesh_{suffix}"
    (paths_offsets, paths_data), points = getattr(_trueform.intersect, func_name)(
        mesh._wrapper
    )

    # 5. WRAP PATHS IN OffsetBlockedArray
    paths = OffsetBlockedArray(paths_offsets, paths_data)

    return paths, points
