"""
neighborhoods function implementation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from .. import _trueform
from .._core import OffsetBlockedArray
from .._dispatch import topology_suffix


def neighborhoods(
    connectivity: OffsetBlockedArray,
    points: np.ndarray,
    radius: float,
    inclusive: bool = False
) -> OffsetBlockedArray:
    """
    Compute radius-based neighborhoods for all vertices.

    For each vertex, computes all vertices reachable via mesh edges where
    the Euclidean distance from the seed vertex is within the specified radius.

    Parameters
    ----------
    connectivity : OffsetBlockedArray
        Vertex connectivity (1-ring) from vertex_link_edges or vertex_link_faces.

    points : np.ndarray
        Vertex positions with shape (n_vertices, dims). Must be float32 or float64.

    radius : float
        Maximum Euclidean distance from seed vertex.

    inclusive : bool, optional
        If True, include the seed vertex in its own neighborhood.
        Default is False.

    Returns
    -------
    OffsetBlockedArray
        Neighborhoods where block i contains the indices of all vertices
        reachable from vertex i within the specified radius.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Triangle mesh
    >>> faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [1.5, 1, 0]], dtype=np.float32)
    >>>
    >>> # Build connectivity
    >>> fm = tf.cell_membership(faces, n_ids=len(points))
    >>> connectivity = tf.vertex_link_faces(faces, fm)
    >>>
    >>> # Compute neighborhoods within radius 1.5
    >>> neighs = tf.neighborhoods(connectivity, points, radius=1.5)
    >>> # neighs[0] contains all vertices within distance 1.5 of vertex 0
    """

    # ===== VALIDATE connectivity =====
    if not isinstance(connectivity, OffsetBlockedArray):
        raise TypeError(
            f"connectivity must be OffsetBlockedArray, "
            f"got {type(connectivity).__name__}"
        )

    # ===== VALIDATE points =====
    if not isinstance(points, np.ndarray):
        raise TypeError(
            f"points must be np.ndarray, got {type(points).__name__}"
        )

    if points.ndim != 2:
        raise ValueError(
            f"points must be 2D array with shape (n, dims), "
            f"got {points.ndim}D array with shape {points.shape}"
        )

    if points.dtype not in (np.float32, np.float64):
        raise TypeError(
            f"points dtype must be float32 or float64, got {points.dtype}. "
            f"Convert with points.astype(np.float32) or points.astype(np.float64)"
        )

    dims = points.shape[1]
    if dims != 3:
        raise ValueError(
            f"points must have 3 dimensions, got {dims}"
        )

    # Ensure C-contiguous
    if not points.flags['C_CONTIGUOUS']:
        points = np.ascontiguousarray(points)

    # ===== VALIDATE radius =====
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")

    # ===== BUILD SUFFIX AND DISPATCH =====
    suffix = topology_suffix(
        connectivity.offsets.dtype,
        real_dtype=points.dtype,
        dims=dims
    )
    func_name = f"make_neighborhoods_{suffix}"
    cpp_func = getattr(_trueform.topology, func_name)

    # Call C++ function
    wrapper = cpp_func(connectivity._wrapper, points, radius, inclusive)

    return OffsetBlockedArray(wrapper.offsets_array(), wrapper.data_array())
