"""
Triangulation of polygons

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import Union, Tuple
import numpy as np
from .. import _trueform
from .._spatial import Mesh
from .._core import OffsetBlockedArray, as_offset_blocked
from .._dispatch import extract_meta, build_suffix, InputMeta


def triangulated(
    data: Union[np.ndarray, Mesh, Tuple[np.ndarray, np.ndarray],
                Tuple[OffsetBlockedArray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangulate polygons into triangles.

    Parameters
    ----------
    data : np.ndarray, Mesh, or tuple
        - np.ndarray shape (N, D): Single polygon with N vertices in D dimensions
        - Mesh: Triangle mesh or dynamic n-gon mesh
        - (faces, points): Tuple with face indices and point coordinates
        - (OffsetBlockedArray, points): Dynamic polygon mesh

    Returns
    -------
    faces : np.ndarray of shape (num_triangles, 3)
        Triangle indices
    points : np.ndarray of shape (num_points, D)
        Vertex coordinates (copied from input)

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Single polygon
    >>> polygon = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float32)
    >>> faces, points = tf.triangulated(polygon)
    >>>
    >>> # Quad mesh
    >>> quads = np.array([[0,1,2,3]], dtype=np.int32)
    >>> pts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float32)
    >>> faces, points = tf.triangulated((quads, pts))
    """
    meta = extract_meta(data)

    # Single polygon (points only)
    if meta.ngon is None:
        suffix = build_suffix(meta)
        func = getattr(_trueform.geometry, f"triangulated_{suffix}")
        return func(data)

    # Triangle mesh - return as-is (no-op)
    if meta.ngon == '3':
        if isinstance(data, Mesh):
            return data.faces.copy(), data.points.copy()
        else:
            indices, points = data
            return indices.copy(), points.copy()

    # Fixed ngon > 3 - convert to dynamic
    if meta.ngon != 'dyn':
        if isinstance(data, Mesh):
            indices = data.faces
            points = data.points
        else:
            indices, points = data
        indices = as_offset_blocked(indices)
        # Update meta to dynamic
        meta = InputMeta(meta.index_dtype, meta.real_dtype, 'dyn', meta.dims)
    else:
        # Already dynamic
        if isinstance(data, Mesh):
            indices = data.faces
            points = data.points
        else:
            indices, points = data

    # Call dynamic C++ function
    suffix = build_suffix(meta)
    func = getattr(_trueform.geometry, f"triangulated_{suffix}")
    return func(indices._wrapper, points)
