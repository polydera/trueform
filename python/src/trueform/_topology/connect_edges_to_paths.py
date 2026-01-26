"""
connect_edges_to_paths() function implementation

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from .. import _trueform
from .._core import OffsetBlockedArray
from .._dispatch import dtype_str


def connect_edges_to_paths(edges: np.ndarray) -> OffsetBlockedArray:
    """
    Connect a collection of edges into continuous paths.

    Takes an arbitrary collection of edges and connects them into paths where
    consecutive edges share vertices. This is a general utility for organizing
    edge sets into connected sequences.

    Parameters
    ----------
    edges : np.ndarray
        Edge array of shape (N, 2) with dtype int32 or int64.
        Each row is [vertex_a, vertex_b] defining an edge.

    Returns
    -------
    OffsetBlockedArray
        Connected paths where each block is a sequence of vertex indices
        forming a continuous path. Paths may be open or closed loops.
        Returns empty OffsetBlockedArray if no edges provided.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Create a set of edges that form paths
    >>> edges = np.array([
    ...     [0, 1],
    ...     [1, 2],
    ...     [2, 3],
    ...     [4, 5],  # Separate path
    ...     [5, 6]
    ... ], dtype=np.int32)
    >>>
    >>> paths = tf.connect_edges_to_paths(edges)
    >>> print(f"Found {len(paths)} paths")
    >>>
    >>> # Iterate over paths
    >>> for i, path in enumerate(paths):
    ...     print(f"Path {i}: {len(path)} vertices")
    ...     # path contains sequence of connected vertex indices

    Notes
    -----
    - Unlike boundary_paths which only works on mesh boundaries, this function
      works on any collection of edges
    - Useful for organizing edges from feature detection, cuts, or general
      graph traversal into connected sequences
    - Edges don't need to be sorted or pre-organized
    """

    # Validate input
    if not isinstance(edges, np.ndarray):
        raise TypeError(
            f"edges must be np.ndarray, got {type(edges).__name__}"
        )

    if edges.ndim != 2:
        raise ValueError(
            f"edges must be 2D array with shape (N, 2), got shape {edges.shape}"
        )

    if edges.shape[1] != 2:
        raise ValueError(
            f"edges must have 2 columns [vertex_a, vertex_b], got {edges.shape[1]} columns"
        )

    if edges.dtype not in (np.int32, np.int64):
        raise TypeError(
            f"edges dtype must be int32 or int64, got {edges.dtype}. "
            f"Convert with edges.astype(np.int32) or edges.astype(np.int64)"
        )

    # Ensure C-contiguous
    if not edges.flags['C_CONTIGUOUS']:
        edges = np.ascontiguousarray(edges)

    # Handle empty input
    if len(edges) == 0:
        empty_offsets = np.array([0], dtype=edges.dtype)
        empty_data = np.array([], dtype=edges.dtype)
        return OffsetBlockedArray(empty_offsets, empty_data)

    # Build suffix and dispatch
    func_name = f"connect_edges_to_paths_{dtype_str(edges.dtype)}"
    cpp_func = getattr(_trueform.topology, func_name)

    # Call C++ function - returns tuple (offsets, data)
    offsets, data = cpp_func(edges)

    return OffsetBlockedArray(offsets, data)
