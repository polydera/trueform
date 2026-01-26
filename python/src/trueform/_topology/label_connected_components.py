"""
label_connected_components() function implementation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Union, Tuple, Optional
from .. import _trueform
from .._core import OffsetBlockedArray
from .._dispatch import connectivity_suffix


def label_connected_components(
    connectivity: Union[np.ndarray, OffsetBlockedArray],
    expected_number_of_components: Optional[int] = None
) -> Tuple[int, np.ndarray]:
    """
    Label connected components in a graph defined by connectivity data.

    Uses a union-find algorithm to identify connected components. The algorithm
    adaptively chooses between parallel and sequential execution based on the
    expected number of components.

    Parameters
    ----------
    connectivity : np.ndarray or OffsetBlockedArray
        Graph connectivity data defining edges between nodes:

        - **np.ndarray**: Fixed-width connectivity, shape (N, K) where each row
          contains K neighbor indices. Use -1 for missing neighbors.
          Must have dtype int32 or int64.

        - **OffsetBlockedArray**: Variable-length connectivity where each block
          contains neighbor indices for one node. Must have dtype int32 or int64.

    expected_number_of_components : int, optional
        Hint for the expected number of components (default: None).

        - When > 500: Uses sequential algorithm (better for many small components)
        - When ≤ 500 or None: Uses parallel algorithm (better for fewer large components)

    Returns
    -------
    num_components : int
        Number of connected components found

    labels : np.ndarray
        Component labels for each node, shape (N,) with dtype int32.
        Labels range from 0 to num_components-1. Nodes with the same label
        belong to the same connected component.

    Raises
    ------
    TypeError
        If connectivity is not np.ndarray or OffsetBlockedArray
        If connectivity dtype is not int32 or int64
        If expected_number_of_components is not an int or None

    ValueError
        If connectivity array has incorrect shape or dimensions

    Notes
    -----
    - Negative indices (e.g., -1) in connectivity are treated as "no neighbor"
    - The algorithm is deterministic: same input always produces same labeling
    - For OffsetBlockedArray, both offsets and data must have matching dtypes

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Fixed-width connectivity (triangular mesh as graph)
    >>> # Each triangle connects to its 3 edge-adjacent neighbors
    >>> connectivity = np.array([
    ...     [1, -1, -1],  # Triangle 0 connects to triangle 1
    ...     [0, 2, -1],   # Triangle 1 connects to triangles 0 and 2
    ...     [1, 3, -1],   # Triangle 2 connects to triangles 1 and 3
    ...     [2, -1, -1]   # Triangle 3 connects to triangle 2
    ... ], dtype=np.int32)
    >>> num_comps, labels = tf.label_connected_components(connectivity)
    >>> print(f"Found {num_comps} component(s)")
    Found 1 component(s)
    >>> print(labels)  # All triangles in same component
    [0 0 0 0]
    >>>
    >>> # Variable-length connectivity using OffsetBlockedArray
    >>> # Node 0: neighbors [1, 2]
    >>> # Node 1: neighbors [0, 3]
    >>> # Node 2: neighbors [0]
    >>> # Node 3: neighbors [1]
    >>> offsets = np.array([0, 2, 4, 5, 6], dtype=np.int32)
    >>> data = np.array([1, 2, 0, 3, 0, 1], dtype=np.int32)
    >>> conn_oba = tf.OffsetBlockedArray(offsets, data)
    >>> num_comps, labels = tf.label_connected_components(conn_oba)
    >>> print(f"Found {num_comps} component(s)")
    Found 1 component(s)
    >>>
    >>> # Disconnected graph
    >>> connectivity = np.array([
    ...     [1, -1],     # Component 0: nodes 0-1
    ...     [0, -1],
    ...     [3, -1],     # Component 1: nodes 2-3
    ...     [2, -1]
    ... ], dtype=np.int32)
    >>> num_comps, labels = tf.label_connected_components(connectivity)
    >>> print(f"Found {num_comps} component(s)")
    Found 2 component(s)
    >>> print(labels)
    [0 0 1 1]
    >>>
    >>> # Large graph with many components - use hint for better performance
    >>> num_comps, labels = tf.label_connected_components(
    ...     connectivity, expected_number_of_components=1000
    ... )
    """

    # ===== VALIDATE expected_number_of_components =====
    if expected_number_of_components is not None:
        if not isinstance(expected_number_of_components, int):
            raise TypeError(
                f"expected_number_of_components must be int or None, "
                f"got {type(expected_number_of_components).__name__}"
            )

    # ===== HANDLE OffsetBlockedArray INPUT =====
    if isinstance(connectivity, OffsetBlockedArray):
        # Validate dtype
        if connectivity.offsets.dtype not in (np.int32, np.int64):
            raise TypeError(
                f"OffsetBlockedArray offsets must be int32 or int64, "
                f"got {connectivity.offsets.dtype}"
            )
        if connectivity.data.dtype not in (np.int32, np.int64):
            raise TypeError(
                f"OffsetBlockedArray data must be int32 or int64, "
                f"got {connectivity.data.dtype}"
            )

        # Build suffix and dispatch
        suffix = connectivity_suffix('offset_blocked', connectivity.offsets.dtype)
        func_name = f"label_connected_components_{suffix}"
        cpp_func = getattr(_trueform.topology, func_name)
        num_components, labels = cpp_func(
            connectivity._wrapper,
            expected_number_of_components
        )

        return num_components, labels

    # ===== HANDLE ndarray INPUT =====
    elif isinstance(connectivity, np.ndarray):
        # Validate shape
        if connectivity.ndim != 2:
            raise ValueError(
                f"connectivity array must be 2D with shape (N, K), "
                f"got {connectivity.ndim}D array with shape {connectivity.shape}"
            )

        # Validate dtype
        if connectivity.dtype not in (np.int32, np.int64):
            raise TypeError(
                f"connectivity dtype must be int32 or int64, got {connectivity.dtype}. "
                f"Convert with connectivity.astype(np.int32) or connectivity.astype(np.int64)"
            )

        # Ensure C-contiguous
        if not connectivity.flags['C_CONTIGUOUS']:
            connectivity = np.ascontiguousarray(connectivity)

        # Build suffix and dispatch
        suffix = connectivity_suffix('ndarray', connectivity.dtype)
        func_name = f"label_connected_components_{suffix}"
        cpp_func = getattr(_trueform.topology, func_name)
        num_components, labels = cpp_func(
            connectivity,
            expected_number_of_components
        )

        return num_components, labels

    else:
        raise TypeError(
            f"Expected np.ndarray or OffsetBlockedArray, "
            f"got {type(connectivity).__name__}"
        )
