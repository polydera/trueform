"""
non_manifold_vertices() function implementation

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from .. import _trueform
from .._spatial import Mesh
from .._dispatch import topology_suffix


def non_manifold_vertices(mesh: Mesh) -> np.ndarray:
    """
    Find non-manifold vertices in a mesh.

    A vertex is non-manifold when the faces around it are not one fan:
    an edge at it carries three or more faces, or its faces fall into
    several fans that meet at the vertex alone (a bowtie). Face winding
    does not enter the verdict, and a vertex no face names is not
    reported.

    Parameters
    ----------
    mesh : Mesh
        The mesh to check for non-manifold vertices. Supports triangular
        meshes (ngon=3) and dynamic meshes with variable polygon sizes.

    Returns
    -------
    np.ndarray
        Array of non-manifold vertex indices with shape (N,), ascending.
        Empty when every vertex's faces are one fan.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Two triangles meeting at a single vertex (a bowtie)
    >>> faces = np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int32)
    >>> points = np.array([
    ...     [0, 0, 0], [1, 0, 0], [0.5, 1, 0], [1.5, 2, 0], [0.5, 2, 0]
    ... ], dtype=np.float32)
    >>> mesh = tf.Mesh(faces, points)
    >>> tf.non_manifold_vertices(mesh)
    array([2], dtype=int32)
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
    func_name = f"non_manifold_vertices_{suffix}"
    cpp_func = getattr(_trueform.topology, func_name)

    if mesh.is_dynamic:
        return cpp_func(mesh._wrapper.faces_array(), fm)
    return cpp_func(faces, fm)
