"""
is_manifold() and is_non_manifold() function implementations

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from .. import _trueform
from .._spatial import Mesh
from .._dispatch import topology_suffix


def is_manifold(mesh: Mesh) -> bool:
    """
    Check if a mesh is manifold.

    A manifold mesh has no non-manifold edges - every edge is shared by
    at most two faces. Non-manifold edges (shared by 3+ faces) indicate
    self-intersections or invalid topology.

    Parameters
    ----------
    mesh : Mesh
        The mesh to check.

    Returns
    -------
    bool
        True if the mesh is manifold, False otherwise.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Two triangles sharing an edge is manifold
    >>> faces = np.array([[0,1,2], [1,3,2]], dtype=np.int32)
    >>> points = np.array([[0,0,0], [1,0,0], [0.5,1,0], [1.5,1,0]], dtype=np.float32)
    >>> mesh = tf.Mesh(faces, points)
    >>> tf.is_manifold(mesh)
    True
    """
    if not isinstance(mesh, Mesh):
        raise TypeError(f"mesh must be Mesh, got {type(mesh).__name__}")

    if not mesh.is_dynamic and mesh.ngon != 3:
        raise ValueError(
            f"mesh must have triangular faces or be dynamic, got {mesh.ngon} vertices per face."
        )

    faces = mesh.faces
    fm = mesh._wrapper.face_membership_array()

    ngon = 'dyn' if mesh.is_dynamic else '3'
    suffix = topology_suffix(faces.dtype, ngon)
    func_name = f"is_manifold_{suffix}"
    cpp_func = getattr(_trueform.topology, func_name)

    if mesh.is_dynamic:
        return cpp_func(mesh._wrapper.faces_array(), fm)
    return cpp_func(faces, fm)


def is_non_manifold(mesh: Mesh) -> bool:
    """
    Check if a mesh has non-manifold edges.

    Returns True if any edge in the mesh is shared by more than two faces.

    Parameters
    ----------
    mesh : Mesh
        The mesh to check.

    Returns
    -------
    bool
        True if the mesh has non-manifold edges, False otherwise.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Three triangles sharing the same edge is non-manifold
    >>> faces = np.array([[0,1,2], [0,1,3], [0,1,4]], dtype=np.int32)
    >>> points = np.array([
    ...     [0,0,0], [1,0,0], [0.5,1,0], [0.5,-1,0], [0.5,0,1]
    ... ], dtype=np.float32)
    >>> mesh = tf.Mesh(faces, points)
    >>> tf.is_non_manifold(mesh)
    True
    """
    return not is_manifold(mesh)
