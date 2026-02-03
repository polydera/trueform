"""
is_closed() and is_open() function implementations

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from .. import _trueform
from .._spatial import Mesh
from .._dispatch import topology_suffix


def is_closed(mesh: Mesh) -> bool:
    """
    Check if a mesh is closed (watertight).

    A closed mesh has no boundary edges - every edge is shared by exactly
    two faces. Closed meshes are watertight and enclose a volume.

    Parameters
    ----------
    mesh : Mesh
        The mesh to check.

    Returns
    -------
    bool
        True if the mesh is closed, False otherwise.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # A tetrahedron is closed
    >>> faces = np.array([[0,1,2], [0,2,3], [0,3,1], [1,3,2]], dtype=np.int32)
    >>> points = np.array([[0,0,0], [1,0,0], [0.5,1,0], [0.5,0.5,1]], dtype=np.float32)
    >>> mesh = tf.Mesh(faces, points)
    >>> tf.is_closed(mesh)
    True
    >>>
    >>> # A single triangle is open
    >>> faces = np.array([[0, 1, 2]], dtype=np.int32)
    >>> points = np.array([[0,0,0], [1,0,0], [0,1,0]], dtype=np.float32)
    >>> mesh = tf.Mesh(faces, points)
    >>> tf.is_closed(mesh)
    False
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
    func_name = f"is_closed_{suffix}"
    cpp_func = getattr(_trueform.topology, func_name)

    if mesh.is_dynamic:
        return cpp_func(mesh._wrapper.faces_array(), fm)
    return cpp_func(faces, fm)


def is_open(mesh: Mesh) -> bool:
    """
    Check if a mesh is open (has boundary edges).

    An open mesh has at least one boundary edge - an edge that belongs
    to only one face.

    Parameters
    ----------
    mesh : Mesh
        The mesh to check.

    Returns
    -------
    bool
        True if the mesh is open, False otherwise.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # A single triangle is open
    >>> faces = np.array([[0, 1, 2]], dtype=np.int32)
    >>> points = np.array([[0,0,0], [1,0,0], [0,1,0]], dtype=np.float32)
    >>> mesh = tf.Mesh(faces, points)
    >>> tf.is_open(mesh)
    True
    """
    return not is_closed(mesh)
