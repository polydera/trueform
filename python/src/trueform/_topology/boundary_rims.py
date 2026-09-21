"""
boundary_rims() function implementation

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import Tuple
import numpy as np
from .. import _trueform
from .._core import OffsetBlockedArray
from .._spatial import Mesh
from .._dispatch import topology_suffix


def boundary_rims(
    mesh: Mesh
) -> Tuple[OffsetBlockedArray, OffsetBlockedArray, np.ndarray]:
    """
    Assemble a mesh's boundary edges into rims.

    Block `i` of `vertices` is rim `i` walked, and block `i` of `faces`
    names the face carrying each of its edges, so rim edge `k` runs from
    vertex `k` to vertex `k + 1` and is carried by face `k` alone. A
    closed rim of `n` vertices has `n` edges, the last running back to
    vertex `0`; an open one has `n - 1`.

    A rim ends where the boundary stops passing straight through, so a
    pinched boundary comes back as its pieces rather than as one figure
    eight. A consistently wound mesh gives rims that run the way its
    faces wind their boundary.

    Parameters
    ----------
    mesh : Mesh
        The mesh to extract boundary rims from. Supports triangular
        meshes (ngon=3) and dynamic meshes with variable polygon sizes.

    Returns
    -------
    vertices : OffsetBlockedArray
        Per rim, its vertex indices in the order it is walked.
    faces : OffsetBlockedArray
        Per rim, the face carrying each of its edges.
    closed : np.ndarray
        Array of shape (R,) with dtype int8: nonzero where rim `i`'s
        last edge runs back to its first vertex.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # A quad of two triangles: one closed rim of four vertices
    >>> faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    >>> points = np.array([
    ...     [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
    ... ], dtype=np.float32)
    >>> mesh = tf.Mesh(faces, points)
    >>>
    >>> vertices, rim_faces, closed = tf.boundary_rims(mesh)
    >>> len(vertices)
    1
    >>> bool(closed[0])
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
    func_name = f"boundary_rims_{suffix}"
    cpp_func = getattr(_trueform.topology, func_name)

    if mesh.is_dynamic:
        vertices, rim_faces, closed = cpp_func(mesh._wrapper.faces_array(), fm)
    else:
        vertices, rim_faces, closed = cpp_func(faces, fm)

    return (
        OffsetBlockedArray(vertices.offsets_array(), vertices.data_array()),
        OffsetBlockedArray(rim_faces.offsets_array(), rim_faces.data_array()),
        closed,
    )
