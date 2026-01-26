"""
Intersection curves between meshes

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Tuple
from .. import _trueform
from .._spatial import Mesh
from .._core import OffsetBlockedArray
from .._dispatch import extract_meta, build_suffix_pair, canonicalize_index_order


def intersection_curves(
    mesh0: Mesh,
    mesh1: Mesh
) -> Tuple[OffsetBlockedArray, np.ndarray]:
    """
    Compute intersection curves between two 3D meshes.

    Finds the curves where two meshes intersect in 3D space.
    Returns paths (as indices into points) and the curve point coordinates.

    Parameters
    ----------
    mesh0 : Mesh
        First 3D mesh with topology
    mesh1 : Mesh
        Second 3D mesh with topology
        Must have same real dtype (float32 or float64) as mesh0

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
        If meshes are not 3D
        If meshes have different real dtypes
    TypeError
        If inputs are not Mesh objects

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>> # Create two intersecting meshes
    >>> faces1, points1 = tf.read_stl("mesh1.stl")
    >>> faces2, points2 = tf.read_stl("mesh2.stl")
    >>> mesh1 = tf.Mesh(faces1, points1)
    >>> mesh2 = tf.Mesh(faces2, points2)
    >>>
    >>> # Compute intersection curves
    >>> paths, points = tf.intersection_curves(mesh1, mesh2)
    >>> print(f"Found {len(paths)} curve(s)")
    >>>
    >>> # Iterate over curves
    >>> for path_ids in paths:
    ...     curve_points = points[path_ids]
    ...     # Process curve (e.g., plot, analyze, etc.)
    """

    # 1. VALIDATE INPUTS ARE MESH OBJECTS
    if not isinstance(mesh0, Mesh):
        raise TypeError(
            f"mesh0 must be a Mesh object, got {type(mesh0).__name__}. "
            f"Topology information is required for intersection curves."
        )

    if not isinstance(mesh1, Mesh):
        raise TypeError(
            f"mesh1 must be a Mesh object, got {type(mesh1).__name__}. "
            f"Topology information is required for intersection curves."
        )

    # 2. VALIDATE BOTH ARE 3D
    if mesh0.dims != 3:
        raise ValueError(
            f"intersection_curves only supports 3D meshes, got mesh0 with {mesh0.dims}D"
        )
    if mesh1.dims != 3:
        raise ValueError(
            f"intersection_curves only supports 3D meshes, got mesh1 with {mesh1.dims}D"
        )

    # 3. VALIDATE REAL DTYPES MATCH
    if mesh0.dtype != mesh1.dtype:
        raise ValueError(
            f"Mesh dtypes must match: mesh0 has {mesh0.dtype}, mesh1 has {mesh1.dtype}. "
            f"Convert both meshes to the same dtype (float32 or float64)."
        )

    # 4. HANDLE INDEX TYPE SYMMETRY
    # C++ only implements: int×int, int×int64, int64×int64
    # If we have int64×int, swap to int×int64
    mesh0, mesh1, _ = canonicalize_index_order(mesh0, mesh1)

    # 5. BUILD SUFFIX FOR C++ FUNCTION
    meta0 = extract_meta(mesh0)
    meta1 = extract_meta(mesh1)
    suffix = build_suffix_pair(meta0, meta1)

    # 6. DISPATCH TO C++
    func_name = f"intersection_curves_mesh_mesh_{suffix}"
    (paths_offsets, paths_data), points = getattr(_trueform.intersect, func_name)(
        mesh0._wrapper, mesh1._wrapper
    )

    # 7. WRAP PATHS IN OffsetBlockedArray
    paths = OffsetBlockedArray(paths_offsets, paths_data)

    return paths, points
