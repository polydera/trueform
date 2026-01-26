"""
boundary_paths() function implementation

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from .. import _trueform
from .._core import OffsetBlockedArray
from .._spatial import Mesh
from .._dispatch import topology_suffix


def boundary_paths(mesh: Mesh) -> OffsetBlockedArray:
    """
    Extract boundary paths from a mesh.

    Connects boundary edges into paths (loops). Each path is a connected
    sequence of boundary edges forming a closed loop around a hole or
    the mesh exterior.

    Parameters
    ----------
    mesh : Mesh
        The mesh to extract boundary paths from. Supports triangular
        meshes (ngon=3) and dynamic meshes with variable polygon sizes.

    Returns
    -------
    OffsetBlockedArray
        Boundary paths where each block is a connected loop of vertex indices.
        Vertex indices reference the original mesh points.
        Returns empty OffsetBlockedArray if mesh has no boundaries.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Create a mesh with a hole
    >>> faces = np.array([
    ...     [0, 1, 2],
    ...     [0, 2, 3]
    ... ], dtype=np.int32)
    >>> points = np.array([
    ...     [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
    ... ], dtype=np.float32)
    >>> mesh = tf.Mesh(faces, points)
    >>>
    >>> paths = tf.boundary_paths(mesh)
    >>> print(f"Found {len(paths)} boundary loops")
    >>>
    >>> # Iterate over boundary loops
    >>> for i, path in enumerate(paths):
    ...     print(f"Loop {i}: {len(path)} vertices")
    ...     # path contains original mesh vertex indices
    """

    # Validate input
    if not isinstance(mesh, Mesh):
        raise TypeError(
            f"mesh must be Mesh, got {type(mesh).__name__}"
        )

    # Validate ngon - only triangles or dynamic
    if not mesh.is_dynamic and mesh.ngon != 3:
        raise ValueError(
            f"mesh must have triangular faces or be dynamic, got {mesh.ngon} vertices per face. "
            f"boundary_paths only supports triangles and dynamic meshes."
        )

    # Get faces and face_membership from mesh
    faces = mesh.faces
    fm = mesh._wrapper.face_membership_array()

    # Build suffix and dispatch
    ngon = 'dyn' if mesh.is_dynamic else '3'
    suffix = topology_suffix(faces.dtype, ngon)

    func_name = f"boundary_paths_{suffix}"
    cpp_func = getattr(_trueform.topology, func_name)

    # Call C++ function - returns offset_blocked_array_wrapper
    # For dynamic, pass the wrapper directly; for fixed ngon, pass numpy array
    if mesh.is_dynamic:
        wrapper = cpp_func(mesh._wrapper.faces_array(), fm)
    else:
        wrapper = cpp_func(faces, fm)

    return OffsetBlockedArray(wrapper.offsets_array(), wrapper.data_array())
