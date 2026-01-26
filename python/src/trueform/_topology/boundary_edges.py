"""
boundary_edges() function implementation

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from .. import _trueform
from .._spatial import Mesh
from .._dispatch import topology_suffix


def boundary_edges(mesh: Mesh) -> np.ndarray:
    """
    Extract boundary edges from a mesh.

    Boundary edges are edges that belong to only one face (i.e., they are
    on the mesh boundary, not shared between two faces).

    Parameters
    ----------
    mesh : Mesh
        The mesh to extract boundary edges from. Supports triangular
        meshes (ngon=3) and dynamic meshes with variable polygon sizes.

    Returns
    -------
    np.ndarray
        Array of boundary edges with shape (N, 2) where N is the number of
        boundary edges. Each row contains two vertex indices defining an edge.
        Returns empty array with shape (0, 2) if mesh has no boundaries.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Create a single triangle (all edges are boundary)
    >>> faces = np.array([[0, 1, 2]], dtype=np.int32)
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    >>> mesh = tf.Mesh(faces, points)
    >>>
    >>> edges = tf.boundary_edges(mesh)
    >>> print(f"Found {len(edges)} boundary edges")
    >>>
    >>> # Check if mesh is watertight (closed)
    >>> if len(edges) == 0:
    ...     print("Mesh is watertight")
    ... else:
    ...     print(f"Mesh has {len(edges)} boundary edges")
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
            f"boundary_edges only supports triangles and dynamic meshes."
        )

    # Get faces and face_membership from mesh
    faces = mesh.faces
    fm = mesh._wrapper.face_membership_array()

    # Build suffix and dispatch
    ngon = 'dyn' if mesh.is_dynamic else '3'
    suffix = topology_suffix(faces.dtype, ngon)
    func_name = f"boundary_edges_{suffix}"
    cpp_func = getattr(_trueform.topology, func_name)

    # Call C++ function - returns ndarray
    # For dynamic, pass the wrapper directly; for fixed ngon, pass numpy array
    if mesh.is_dynamic:
        return cpp_func(mesh._wrapper.faces_array(), fm)
    else:
        return cpp_func(faces, fm)
