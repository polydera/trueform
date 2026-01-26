"""
boundary_curves() function implementation

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Tuple
from .._core import OffsetBlockedArray
from .._spatial import Mesh
from .boundary_paths import boundary_paths


def boundary_curves(mesh: Mesh) -> Tuple[OffsetBlockedArray, np.ndarray]:
    """
    Extract boundary curves from a mesh as standalone geometry.

    Returns the boundary as a self-contained structure with its own points
    array and connectivity reindexed from 0 to N. This is useful when you
    want to work with the boundary as independent geometry.

    Parameters
    ----------
    mesh : Mesh
        The mesh to extract boundary curves from. Supports triangular
        meshes (ngon=3) and dynamic meshes with variable polygon sizes.

    Returns
    -------
    paths : OffsetBlockedArray
        Boundary paths where each block is a connected loop with indices
        referencing the returned points array (0 to N-1).
    points : np.ndarray
        Array of boundary points with shape (N, dims). Contains only the
        vertices that lie on the boundary.

    Examples
    --------
    >>> import trueform as tf
    >>> # Load mesh with boundaries
    >>> faces, points = tf.read_stl("mesh_with_holes.stl")
    >>> mesh = tf.Mesh(faces, points)
    >>>
    >>> # Get boundary curves
    >>> paths, curve_points = tf.boundary_curves(mesh)
    >>> print(f"Found {len(paths)} boundary loop(s)")
    >>>
    >>> # Iterate over boundary loops
    >>> for path_ids in paths:
    ...     loop_points = curve_points[path_ids]
    ...     # Process loop (e.g., plot, measure, etc.)

    See Also
    --------
    boundary_paths : Returns paths with original mesh vertex indices.
    boundary_edges : Returns boundary edges without connecting into paths.
    """

    # Validate input
    if not isinstance(mesh, Mesh):
        raise TypeError(
            f"mesh must be Mesh, got {type(mesh).__name__}"
        )

    # Get boundary paths with original vertex indices
    paths = boundary_paths(mesh)

    # Handle empty boundary case
    if len(paths.data) == 0:
        empty_points = np.empty((0, mesh.dims), dtype=mesh.points.dtype)
        empty_offsets = np.array([0], dtype=paths.offsets.dtype)
        empty_data = np.array([], dtype=paths.data.dtype)
        return OffsetBlockedArray(empty_offsets, empty_data), empty_points

    # Get unique vertices and inverse mapping using np.unique
    unique_ids, inverse = np.unique(paths.data, return_inverse=True)

    # Extract just those points
    boundary_points = mesh.points[unique_ids]

    # Create new OffsetBlockedArray with remapped indices (0 to N-1)
    remapped_paths = OffsetBlockedArray(
        paths.offsets,
        inverse.astype(paths.data.dtype)
    )

    return remapped_paths, boundary_points
