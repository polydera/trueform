"""
Generalized winding number of a mesh at a point query

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


def winding_number(mesh: Any, query: Any, beta: float = 2.0):
    """
    Compute the generalized winding number of a mesh at a point query.

    Approximately 1 inside a closed surface, 0 outside, and a graceful
    fractional value for open sheets and soups (Barill et al., fast winding
    numbers). Near the surface the value crosses 0.5 continuously, so any
    threshold on it is a float test — a closed input's exact inside/outside
    belongs to the exact predicates, this query to the inputs they refuse.

    The number is read off the winding moments of the mesh's spatial tree:
    far nodes answer by their stored expansion, near leaves by exact solid
    angles. The moments build on first query and are cached with the tree, so
    ``mesh.build_winding_moments()`` only moves when that cost is paid.

    Parameters
    ----------
    mesh : Mesh
        The 3D mesh to measure against. A transformation on the mesh is
        honored — the query is moved into the mesh's own frame.
    query : Point or array-like
        A Point (single or batch), or array-like of shape ``(3,)`` or
        ``(N, 3)`` — array-likes are converted to the mesh's dtype.
    beta : float, default 2.0
        Accuracy knob: a node answers by its stored far-field expansion only
        when the query is farther than ``beta`` times the node's far-field
        radius. Larger is more accurate and descends deeper; the exact
        solid-angle sum is the limit.

    Returns
    -------
    float or np.ndarray
        The winding number. Scalar for a single query, shape ``(N,)`` for a
        batch. Always float64: the number is dimensionless, not a coordinate,
        and the query answers in double whatever the mesh's dtype.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>> mesh = tf.Mesh(*tf.make_sphere_mesh(1.0))
    >>> round(tf.winding_number(mesh, [0.0, 0.0, 0.0]))
    1
    >>> round(tf.winding_number(mesh, [3.0, 0.0, 0.0]))
    0
    >>> queries = tf.Point(np.array([[0, 0, 0], [3, 0, 0]], dtype=np.float32))
    >>> np.round(tf.winding_number(mesh, queries))
    array([1., 0.])
    """
    from . import Mesh

    if not isinstance(mesh, Mesh):
        raise TypeError(
            f"winding_number takes a Mesh and a point query, got "
            f"{type(mesh).__name__}")

    if mesh.dims != 3:
        raise ValueError(
            f"winding_number only supports 3D meshes, got {mesh.dims}D")

    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError(
            f"beta must be finite and positive, got {beta}")

    if isinstance(query, Primitive):
        if not isinstance(query, Point):
            raise TypeError(
                f"winding_number takes a point query, got "
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
        if coords.ndim == 2 and coords.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        query = Point(coords)

    meta = extract_meta(mesh)
    suffix = build_suffix(meta)
    fn = getattr(_trueform.spatial, f"winding_number_{meta.form_name}_fp_{suffix}")
    return fn(mesh._wrapper, query._wrapper, beta)
