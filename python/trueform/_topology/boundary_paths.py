"""
boundary_paths() function implementation

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

import numpy as np
from .. import _trueform
from .._core import OffsetBlockedArray
from .._spatial import Mesh


def boundary_paths(mesh: Mesh) -> OffsetBlockedArray:
    """
    Extract boundary paths from a mesh.

    Connects boundary edges into paths (loops). Each path is a connected
    sequence of boundary edges forming a closed loop around a hole or
    the mesh exterior.

    Parameters
    ----------
    mesh : Mesh
        The mesh to extract boundary paths from.

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

    # Validate ngon - only triangles and quads
    ngon = mesh.ngon
    if ngon not in (3, 4):
        raise ValueError(
            f"mesh must have triangular or quad faces, got {ngon} vertices per face. "
            f"Boundary detection is only defined for triangles and quads."
        )

    # Get faces and face_membership from mesh
    faces = mesh.faces
    fm = mesh._wrapper.face_membership_array()

    # Build suffix and dispatch
    dtype_str = 'int' if faces.dtype == np.int32 else 'int64'
    suffix = f"{dtype_str}_{ngon}"

    func_name = f"boundary_paths_{suffix}"
    cpp_func = getattr(_trueform.topology, func_name)

    # Call C++ function - returns offset_blocked_array_wrapper
    wrapper = cpp_func(faces, fm)

    return OffsetBlockedArray(wrapper.offsets_array(), wrapper.data_array())
