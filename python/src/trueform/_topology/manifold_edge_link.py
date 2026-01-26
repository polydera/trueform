"""
manifold_edge_link() function implementation

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


def manifold_edge_link(
    cells: Union[np.ndarray, OffsetBlockedArray],
    cell_membership: OffsetBlockedArray
) -> Union[np.ndarray, OffsetBlockedArray]:
    """
    Compute manifold edge links for faces.

    For each edge of each face, finds the adjacent face sharing that edge
    (assuming manifold mesh where each edge is shared by at most 2 faces).

    Parameters
    ----------
    cells : np.ndarray or OffsetBlockedArray
        Face connectivity:
        - np.ndarray with shape (N, V) where N is number of faces and V is
          vertices per face (3 for triangles). Must have dtype int32 or int64.
        - OffsetBlockedArray for dynamic meshes with variable polygon sizes.

    cell_membership : OffsetBlockedArray
        Cell membership structure mapping vertices to faces. Can be computed
        using cell_membership().

    Returns
    -------
    np.ndarray or OffsetBlockedArray
        Manifold edge link with same structure as cells:
        - np.ndarray of shape (N, V) for numpy input
        - OffsetBlockedArray for dynamic input
        Entry [i, j] is the index of the face adjacent to face i across edge j.
        Special values:
        - >= 0: index of adjacent face
        - -1: boundary edge (no adjacent face)
        - -2: non-manifold edge (shared by more than 2 faces)
        - -3: non-manifold representative

    Raises
    ------
    TypeError
        If cells has wrong type/dtype, or cell_membership is not OffsetBlockedArray
    ValueError
        If cells has wrong shape or V is not 3 (for numpy arrays)

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Two adjacent triangles sharing edge (1, 2)
    >>> cells = np.array([
    ...     [0, 1, 2],
    ...     [1, 3, 2]
    ... ], dtype=np.int32)
    >>> n_ids = 4
    >>>
    >>> membership = tf.cell_membership(cells, n_ids)
    >>> edge_link = tf.manifold_edge_link(cells, membership)
    >>> # edge_link[0, 1] = 1  # face 0, edge 1 (1-2) is adjacent to face 1
    >>> # edge_link[1, 2] = 0  # face 1, edge 2 (2-1) is adjacent to face 0
    """

    # ===== VALIDATE cell_membership =====
    if not isinstance(cell_membership, OffsetBlockedArray):
        raise TypeError(
            f"cell_membership must be OffsetBlockedArray, "
            f"got {type(cell_membership).__name__}"
        )

    # ===== Handle OffsetBlockedArray (dynamic) =====
    if isinstance(cells, OffsetBlockedArray):
        suffix = topology_suffix(cells.dtype, 'dyn')
        func_name = f"compute_manifold_edge_link_{suffix}"
        cpp_func = getattr(_trueform.topology, func_name)

        wrapper = cpp_func(cells._wrapper, cell_membership._wrapper)
        return OffsetBlockedArray(wrapper.offsets_array(), wrapper.data_array())

    # ===== VALIDATE numpy array cells =====
    if not isinstance(cells, np.ndarray):
        raise TypeError(
            f"cells must be np.ndarray or OffsetBlockedArray, got {type(cells).__name__}"
        )

    if cells.ndim != 2:
        raise ValueError(
            f"cells must be 2D array with shape (N, V), "
            f"got {cells.ndim}D array with shape {cells.shape}"
        )

    # Validate dtype
    if cells.dtype not in (np.int32, np.int64):
        raise TypeError(
            f"cells dtype must be int32 or int64, got {cells.dtype}. "
            f"Convert with cells.astype(np.int32) or cells.astype(np.int64)"
        )

    # Validate ngon (V) - only triangles for numpy arrays
    ngon = cells.shape[1]
    if ngon != 3:
        raise ValueError(
            f"cells must have 3 vertices per face, got {ngon}. "
            f"For variable-size polygons, use OffsetBlockedArray."
        )

    # Check dtype matches
    if cell_membership.offsets.dtype != cells.dtype:
        raise TypeError(
            f"cell_membership dtype ({cell_membership.offsets.dtype}) must match "
            f"cells dtype ({cells.dtype})"
        )

    # Ensure C-contiguous
    if not cells.flags['C_CONTIGUOUS']:
        cells = np.ascontiguousarray(cells)

    # ===== BUILD SUFFIX AND DISPATCH =====
    suffix = topology_suffix(cells.dtype, str(ngon))
    func_name = f"compute_manifold_edge_link_{suffix}"
    cpp_func = getattr(_trueform.topology, func_name)

    # Call C++ function - returns ndarray directly
    result = cpp_func(cells, cell_membership._wrapper)

    return result
