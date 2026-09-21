"""
Dihedral angle computation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import Union, Tuple
import numpy as np
from .. import _trueform
from .._spatial import Mesh
from .._core import OffsetBlockedArray
from .._dispatch import ensure_mesh, extract_meta, build_suffix


def dihedral_angles(
    data: Union[Mesh, Tuple[np.ndarray, np.ndarray], Tuple[OffsetBlockedArray, np.ndarray]]
):
    """
    Measure every edge two faces of a mesh share.

    One entry per undirected edge two faces share. The angle is between
    the face normals, in radians, so a flat surface reads 0. A boundary
    or non-manifold edge joins no pair of faces and turns through no
    angle, so it is not stated.

    Parameters
    ----------
    data : Mesh or tuple
        - Mesh: Mesh object
        - (faces, points): Tuple with face indices and point coordinates
        - (OffsetBlockedArray, points): Dynamic polygon mesh

    Returns
    -------
    edges : np.ndarray of shape (num_edges, 2)
        Vertex index pairs of the shared edges, each pair ascending.
    angles : np.ndarray of shape (num_edges,)
        The angle each edge turns through, in radians, aligned with
        `edges`.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Two coplanar triangles: their shared edge is flat
    >>> faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    >>> points = np.array(
    ...     [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
    ... )
    >>> mesh = tf.Mesh(faces, points)
    >>>
    >>> edges, angles = tf.dihedral_angles(mesh)
    >>> edges
    array([[1, 2]], dtype=int32)
    >>> angles
    array([0.], dtype=float32)
    """
    mesh = ensure_mesh(data, dims=3)
    meta = extract_meta(mesh)
    suffix = build_suffix(meta)
    func = getattr(_trueform.geometry, f"dihedral_angles_{suffix}")
    return func(mesh._wrapper)
