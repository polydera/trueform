"""
make_tube_mesh() function implementation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import Tuple, Union

import numpy as np

from .. import _trueform
from .._core import OffsetBlockedArray
from .._dispatch import dtype_str


def make_tube_mesh(
    data: Union[np.ndarray, Tuple[OffsetBlockedArray, np.ndarray]],
    radius: float,
    *,
    radial_segments: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a triangular tube mesh around one or more 3D polyline curves.

    Closed loops are auto-detected (first ≈ last vertex).

    Parameters
    ----------
    data : np.ndarray or (OffsetBlockedArray, np.ndarray) tuple
        - ``np.ndarray`` shape ``(N, 3)``: a single polyline.  Each row is a
          vertex.  Dtype must be ``float32`` or ``float64``.
        - ``(paths, points)`` tuple: many polylines.  ``paths`` is an
          ``OffsetBlockedArray`` of vertex indices into ``points``
          (e.g. from ``tf.connect_edges_to_paths`` paired with the source
          mesh's points).  Indices dtype ``int32`` or ``int64``; points
          dtype ``float32`` or ``float64``.
    radius : float
        Tube radius.
    radial_segments : int, default 8
        Number of vertices per cross-section ring.

    Returns
    -------
    faces : np.ndarray of shape (n_triangles, 3), dtype int32
    points : np.ndarray of shape (n_vertices, 3)
        Vertex coordinates, same dtype as the input points.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Single polyline
    >>> curve = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float64)
    >>> faces, points = tf.make_tube_mesh(curve, radius=0.05)
    >>>
    >>> # NM-edge paths from a non-manifold arrangement
    >>> nm = tf.non_manifold_edges(mesh)
    >>> paths = tf.connect_edges_to_paths(nm)
    >>> faces, points = tf.make_tube_mesh((paths, mesh.points), radius=0.01)
    """
    # Single polyline (just points)
    if isinstance(data, np.ndarray):
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(
                f"make_tube_mesh: points must have shape (N, 3), "
                f"got {data.shape}"
            )
        if data.dtype not in (np.float32, np.float64):
            raise TypeError(
                f"make_tube_mesh: points dtype must be float32 or float64, "
                f"got {data.dtype}"
            )
        suffix = f"{dtype_str(data.dtype)}3d"
        cpp_func = getattr(_trueform.geometry, f"make_tube_mesh_{suffix}")
        return cpp_func(data, float(data.dtype.type(radius)),
                        int(radial_segments))

    # Multi-curve (paths, points)
    if isinstance(data, tuple) and len(data) == 2:
        paths, points = data
        if not isinstance(paths, OffsetBlockedArray):
            raise TypeError(
                f"make_tube_mesh: paths must be OffsetBlockedArray, "
                f"got {type(paths).__name__}"
            )
        if not isinstance(points, np.ndarray):
            raise TypeError(
                f"make_tube_mesh: points must be np.ndarray, "
                f"got {type(points).__name__}"
            )
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"make_tube_mesh: points must have shape (M, 3), "
                f"got {points.shape}"
            )
        if paths.dtype not in (np.int32, np.int64):
            raise TypeError(
                f"make_tube_mesh: paths dtype must be int32 or int64, "
                f"got {paths.dtype}"
            )
        if points.dtype not in (np.float32, np.float64):
            raise TypeError(
                f"make_tube_mesh: points dtype must be float32 or float64, "
                f"got {points.dtype}"
            )
        suffix = (
            f"{dtype_str(paths.dtype)}dyn{dtype_str(points.dtype)}3d"
        )
        cpp_func = getattr(_trueform.geometry, f"make_tube_mesh_{suffix}")
        return cpp_func(paths._wrapper, points,
                        float(points.dtype.type(radius)),
                        int(radial_segments))

    raise TypeError(
        f"make_tube_mesh: data must be a (N, 3) ndarray or "
        f"(OffsetBlockedArray, points) tuple, got {type(data).__name__}"
    )
