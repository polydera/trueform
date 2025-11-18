"""
non_manifold_edges() function implementation

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

import numpy as np
from .. import _trueform
from .._spatial import Mesh


def non_manifold_edges(mesh: Mesh) -> np.ndarray:
    """
    Find non-manifold edges in a mesh.

    Non-manifold edges are edges that are shared by more than two faces.
    In a valid manifold mesh, each edge should be shared by at most two faces.

    Parameters
    ----------
    mesh : Mesh
        The mesh to check for non-manifold edges.

    Returns
    -------
    np.ndarray
        Array of non-manifold edges with shape (N, 2) where N is the number
        of non-manifold edges. Each row contains two vertex indices defining
        an edge. Returns empty array with shape (0, 2) if mesh is manifold.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> mesh = tf.Mesh(faces, points)
    >>>
    >>> # Check for non-manifold edges
    >>> nm_edges = tf.non_manifold_edges(mesh)
    >>>
    >>> if len(nm_edges) == 0:
    ...     print("Mesh is manifold")
    ... else:
    ...     print(f"Found {len(nm_edges)} non-manifold edges")
    ...     for edge in nm_edges:
    ...         print(f"  Edge: {edge[0]} - {edge[1]}")

    Notes
    -----
    Non-manifold edges commonly occur when:
    - Multiple mesh components share an edge (T-junctions)
    - Mesh was created by merging separate pieces incorrectly
    - Boolean operations produced degenerate geometry

    These edges should typically be fixed before using the mesh in
    algorithms that require manifold input (e.g., boolean operations).
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
            f"Non-manifold detection is only defined for triangles and quads."
        )

    # Get faces and face_membership from mesh
    faces = mesh.faces
    fm = mesh._wrapper.face_membership_array()

    # Build suffix and dispatch
    dtype_str = 'int' if faces.dtype == np.int32 else 'int64'
    suffix = f"{dtype_str}_{ngon}"

    func_name = f"non_manifold_edges_{suffix}"
    cpp_func = getattr(_trueform.topology, func_name)

    # Call C++ function - returns ndarray
    return cpp_func(faces, fm)
