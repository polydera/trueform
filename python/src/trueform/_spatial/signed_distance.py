"""
Signed distance from a mesh surface

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Any
from .. import _trueform
from .._primitives.primitive import Primitive
from .._primitives import Point
from .._dispatch import extract_meta, build_suffix


def signed_distance(obj0: Any, obj1: Any):
    """
    Compute the signed distance between a mesh surface and a point query.

    The magnitude is the Euclidean distance to the closest point on the
    surface; the sign follows the pseudonormal convention (Baerentzen &
    Aanes): negative inside the surface, positive outside. Either argument
    order is accepted. Supports batch queries: a batched Point or an
    ``(N, 3)`` array broadcasts to per-point results.

    Parameters
    ----------
    obj0, obj1 : Mesh and point query, in either order
        The mesh target (3D, with outward-oriented faces) and the query:
        a Point (single or batch), or array-like of shape ``(3,)`` or
        ``(N, 3)`` — array-likes are converted to the mesh's dtype.

    Returns
    -------
    float or np.ndarray
        Signed distance (negative inside, positive outside). Scalar for a
        single query, shape ``(N,)`` in the mesh's dtype for a batch.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>> mesh = tf.Mesh(faces, points)  # a closed cube
    >>> tf.signed_distance(mesh, [0.0, 0.0, 0.0])
    -0.5
    >>> queries = tf.Point(np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float32))
    >>> tf.signed_distance(mesh, queries)
    array([-0.5,  1.5], dtype=float32)
    """
    from . import Mesh

    if isinstance(obj0, Mesh):
        mesh, query = obj0, obj1
    elif isinstance(obj1, Mesh):
        mesh, query = obj1, obj0
    else:
        raise TypeError(
            f"signed_distance requires a Mesh, got: "
            f"{type(obj0).__name__}, {type(obj1).__name__}")

    if mesh.dims != 3:
        raise ValueError(
            f"signed_distance only supports 3D meshes, got {mesh.dims}D")

    if isinstance(query, Primitive):
        if not isinstance(query, Point):
            raise TypeError(
                f"signed_distance takes a point query, got "
                f"{type(query).__name__}")
        if query.dims != mesh.dims:
            raise ValueError(
                f"Dimension mismatch: query has {query.dims}D, mesh has "
                f"{mesh.dims}D")
        if query.dtype != mesh.dtype:
            raise TypeError(
                f"Dtype mismatch: query has {query.dtype}, mesh has "
                f"{mesh.dtype}")
    else:
        coords = np.asarray(query, dtype=mesh.dtype)
        if coords.ndim not in (1, 2) or coords.shape[-1] != 3:
            raise ValueError(
                f"query must have shape (3,) or (N, 3), got {coords.shape}")
        query = Point(coords)

    meta = extract_meta(mesh)
    suffix = build_suffix(meta)
    fn = getattr(_trueform.spatial, f"signed_distance_{meta.form_name}_fp_{suffix}")
    return fn(mesh._wrapper, query._wrapper)
