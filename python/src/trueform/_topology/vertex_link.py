"""
vertex_link functions implementation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Union
from .. import _trueform
from .._core import OffsetBlockedArray
from .._dispatch import topology_suffix


def vertex_link_edges(
    edges: np.ndarray,
    n_ids: int
) -> OffsetBlockedArray:
    """
    Compute vertex links from edge connectivity.

    For each vertex, finds all other vertices that share an edge with it.

    Parameters
    ----------
    edges : np.ndarray
        Edge connectivity array of shape (N, 2) where N is number of edges.
        Must have dtype int32 or int64.

    n_ids : int
        Number of unique vertex IDs. This is typically the number of points.

    Returns
    -------
    OffsetBlockedArray
        Vertex link structure where block i contains the indices of all vertices
        connected to vertex i by an edge.

    Raises
    ------
    TypeError
        If edges is not np.ndarray or has wrong dtype
    ValueError
        If edges has wrong shape

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Edge connectivity (path: 0-1-2-3)
    >>> edges = np.array([
    ...     [0, 1],
    ...     [1, 2],
    ...     [2, 3]
    ... ], dtype=np.int32)
    >>> n_ids = 4
    >>>
    >>> vl = tf.vertex_link_edges(edges, n_ids)
    >>> # vl[0] = [1]        # vertex 0 connects to vertex 1
    >>> # vl[1] = [0, 2]     # vertex 1 connects to vertices 0, 2
    >>> # vl[2] = [1, 3]     # vertex 2 connects to vertices 1, 3
    >>> # vl[3] = [2]        # vertex 3 connects to vertex 2
    """

    # ===== VALIDATE edges =====
    if not isinstance(edges, np.ndarray):
        raise TypeError(
            f"edges must be np.ndarray, got {type(edges).__name__}"
        )

    if edges.ndim != 2:
        raise ValueError(
            f"edges must be 2D array with shape (N, 2), "
            f"got {edges.ndim}D array with shape {edges.shape}"
        )

    # Validate shape
    if edges.shape[1] != 2:
        raise ValueError(
            f"edges must have 2 vertices per edge, got {edges.shape[1]}"
        )

    # Validate dtype
    if edges.dtype not in (np.int32, np.int64):
        raise TypeError(
            f"edges dtype must be int32 or int64, got {edges.dtype}. "
            f"Convert with edges.astype(np.int32) or edges.astype(np.int64)"
        )

    # Ensure C-contiguous
    if not edges.flags['C_CONTIGUOUS']:
        edges = np.ascontiguousarray(edges)

    # ===== BUILD SUFFIX AND DISPATCH =====
    suffix = topology_suffix(edges.dtype, '2')
    func_name = f"compute_vertex_link_{suffix}"
    cpp_func = getattr(_trueform.topology, func_name)

    # Call C++ function - returns offset_blocked_array_wrapper
    wrapper = cpp_func(edges, n_ids)

    return OffsetBlockedArray(wrapper.offsets_array(), wrapper.data_array())


def vertex_link_faces(
    faces: Union[np.ndarray, OffsetBlockedArray],
    cell_membership: OffsetBlockedArray
) -> OffsetBlockedArray:
    """
    Compute vertex links from face connectivity.

    For each vertex, finds all other vertices that share a face with it.

    Parameters
    ----------
    faces : np.ndarray or OffsetBlockedArray
        Face connectivity:
        - np.ndarray with shape (N, V) where N is number of faces and V is
          vertices per face (3 for triangles). Must have dtype int32 or int64.
        - OffsetBlockedArray for dynamic meshes with variable polygon sizes.

    cell_membership : OffsetBlockedArray
        Cell membership structure mapping vertices to faces. Can be computed
        using cell_membership().

    Returns
    -------
    OffsetBlockedArray
        Vertex link structure where block i contains the indices of all vertices
        that share at least one face with vertex i.

    Raises
    ------
    TypeError
        If faces has wrong type/dtype, or cell_membership is not OffsetBlockedArray
    ValueError
        If faces has wrong shape or V is not 3 (for numpy arrays)

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Triangle mesh
    >>> faces = np.array([
    ...     [0, 1, 2],
    ...     [1, 3, 2]
    ... ], dtype=np.int32)
    >>> n_ids = 4
    >>>
    >>> membership = tf.cell_membership(faces, n_ids)
    >>> vl = tf.vertex_link_faces(faces, membership)
    >>> # vl[0] = [1, 2]        # vertex 0 connects to vertices 1, 2
    >>> # vl[1] = [0, 2, 3]     # vertex 1 connects to vertices 0, 2, 3
    """

    # ===== VALIDATE cell_membership =====
    if not isinstance(cell_membership, OffsetBlockedArray):
        raise TypeError(
            f"cell_membership must be OffsetBlockedArray, "
            f"got {type(cell_membership).__name__}"
        )

    # ===== Handle OffsetBlockedArray (dynamic) =====
    if isinstance(faces, OffsetBlockedArray):
        suffix = topology_suffix(faces.dtype, 'dyn')
        func_name = f"compute_vertex_link_{suffix}"
        cpp_func = getattr(_trueform.topology, func_name)

        wrapper = cpp_func(faces._wrapper, cell_membership._wrapper)
        return OffsetBlockedArray(wrapper.offsets_array(), wrapper.data_array())

    # ===== VALIDATE numpy array faces =====
    if not isinstance(faces, np.ndarray):
        raise TypeError(
            f"faces must be np.ndarray or OffsetBlockedArray, got {type(faces).__name__}"
        )

    if faces.ndim != 2:
        raise ValueError(
            f"faces must be 2D array with shape (N, V), "
            f"got {faces.ndim}D array with shape {faces.shape}"
        )

    # Validate dtype
    if faces.dtype not in (np.int32, np.int64):
        raise TypeError(
            f"faces dtype must be int32 or int64, got {faces.dtype}. "
            f"Convert with faces.astype(np.int32) or faces.astype(np.int64)"
        )

    # Validate ngon (V) - only triangles for numpy arrays
    ngon = faces.shape[1]
    if ngon != 3:
        raise ValueError(
            f"faces must have 3 vertices per face, got {ngon}. "
            f"For variable-size polygons, use OffsetBlockedArray."
        )

    # Check dtype matches
    if cell_membership.offsets.dtype != faces.dtype:
        raise TypeError(
            f"cell_membership dtype ({cell_membership.offsets.dtype}) must match "
            f"faces dtype ({faces.dtype})"
        )

    # Ensure C-contiguous
    if not faces.flags['C_CONTIGUOUS']:
        faces = np.ascontiguousarray(faces)

    # ===== BUILD SUFFIX AND DISPATCH =====
    suffix = topology_suffix(faces.dtype, str(ngon))
    func_name = f"compute_vertex_link_{suffix}"
    cpp_func = getattr(_trueform.topology, func_name)

    # Call C++ function - returns offset_blocked_array_wrapper
    wrapper = cpp_func(faces, cell_membership._wrapper)

    return OffsetBlockedArray(wrapper.offsets_array(), wrapper.data_array())
