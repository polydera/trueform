"""
split_non_manifold_vertices() function implementation

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import Tuple
import numpy as np
from .. import _trueform
from .._core import OffsetBlockedArray
from .._dispatch import InputMeta, build_suffix
from .._spatial import Mesh


def split_non_manifold_vertices(mesh: Mesh) -> Tuple[Mesh, np.ndarray]:
    """
    Give every fan at a vertex a vertex of its own.

    A vertex whose faces walk as several fans (a bowtie) keeps its index
    on the fan holding its smallest face; every other fan takes a fresh
    index carrying the same coordinates, and the fan's corners are
    rewired onto it. Faces keep their ids, their arity and their
    winding, and the points are the input's followed by the minted
    copies.

    An edge three or more faces carry is crossed by no fan, and
    separating the fans it holds apart would tear that edge into
    boundary copies. That is another verb's work: a vertex any of whose
    edges carries 3+ faces is left exactly as it was, and
    `non_manifold_vertices` still names it. This one separates only the
    fans that come apart without cutting an edge.

    Parameters
    ----------
    mesh : Mesh
        The mesh to split. Supports triangular meshes (ngon=3) and
        dynamic meshes with variable polygon sizes.

    Returns
    -------
    split : Mesh
        The separated mesh, with the same face-index and point dtypes
        as the input.
    point_map : np.ndarray
        Array of shape (P,): for each output point the input point it
        copies; an original point maps to itself.

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
    >>>
    >>> split, point_map = tf.split_non_manifold_vertices(mesh)
    >>> split.number_of_points
    6
    >>> point_map
    array([0, 1, 2, 3, 4, 2], dtype=int32)
    >>> tf.is_manifold(split)
    True
    """
    if not isinstance(mesh, Mesh):
        raise TypeError(f"mesh must be Mesh, got {type(mesh).__name__}")

    if not mesh.is_dynamic and mesh.ngon != 3:
        raise ValueError(
            f"mesh must have triangular faces or be dynamic, got {mesh.ngon} vertices per face."
        )

    ngon = "dyn" if mesh.is_dynamic else str(mesh.ngon)
    meta = InputMeta(mesh.faces.dtype, mesh.dtype, ngon, mesh.dims)
    suffix = build_suffix(meta)

    func = getattr(_trueform.topology, f"split_non_manifold_vertices_{suffix}")

    faces, points, point_map = func(mesh._wrapper)
    if mesh.is_dynamic:
        faces = OffsetBlockedArray(faces[0], faces[1])
    return Mesh(faces, points), point_map
