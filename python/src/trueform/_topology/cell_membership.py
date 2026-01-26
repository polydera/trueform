"""
cell_membership() function implementation

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


def cell_membership(
    cells: Union[np.ndarray, OffsetBlockedArray],
    n_ids: int
) -> OffsetBlockedArray:
    """
    Compute cell membership for vertices in a connectivity array.

    Maps each vertex ID to the cells (edges, triangles, or polygons) that contain it.

    Parameters
    ----------
    cells : np.ndarray or OffsetBlockedArray
        Cell connectivity array:
        - np.ndarray with shape (N, V) where N is number of cells and V is
          vertices per cell (2 for edges, 3 for triangles). Must have dtype
          int32 or int64.
        - OffsetBlockedArray for dynamic meshes with variable polygon sizes.

    n_ids : int
        Number of unique vertex IDs in the connectivity. This is typically the
        number of points in the mesh.

    Returns
    -------
    OffsetBlockedArray
        Cell membership structure where block i contains the indices of all cells
        that contain vertex i.

    Raises
    ------
    TypeError
        If cells is not np.ndarray or OffsetBlockedArray, or has wrong dtype
    ValueError
        If cells has wrong shape or V is not 2 or 3 (for numpy arrays)

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Triangle connectivity
    >>> cells = np.array([
    ...     [0, 1, 2],
    ...     [1, 2, 3],
    ...     [2, 3, 4]
    ... ], dtype=np.int32)
    >>> n_ids = 5  # vertices 0-4
    >>>
    >>> membership = tf.cell_membership(cells, n_ids)
    >>> # membership[0] = [0]        # vertex 0 is in cell 0
    >>> # membership[1] = [0, 1]     # vertex 1 is in cells 0, 1
    >>> # membership[2] = [0, 1, 2]  # vertex 2 is in cells 0, 1, 2
    """

    # ===== Handle OffsetBlockedArray (dynamic) =====
    if isinstance(cells, OffsetBlockedArray):
        suffix = topology_suffix(cells.dtype, 'dyn')
        func_name = f"compute_cell_membership_{suffix}"
        cpp_func = getattr(_trueform.topology, func_name)

        # Call C++ function with the wrapper
        wrapper = cpp_func(cells._wrapper, n_ids)

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

    # Validate ngon (V) - only 2 or 3 for numpy arrays
    ngon = cells.shape[1]
    if ngon not in (2, 3):
        raise ValueError(
            f"cells must have 2 or 3 vertices per cell, got {ngon}. "
            f"For variable-size polygons, use OffsetBlockedArray."
        )

    # Ensure C-contiguous
    if not cells.flags['C_CONTIGUOUS']:
        cells = np.ascontiguousarray(cells)

    # ===== BUILD SUFFIX AND DISPATCH =====
    suffix = topology_suffix(cells.dtype, str(ngon))
    func_name = f"compute_cell_membership_{suffix}"
    cpp_func = getattr(_trueform.topology, func_name)

    # Call C++ function - returns wrapper
    wrapper = cpp_func(cells, n_ids)

    # Extract arrays and create OffsetBlockedArray
    offsets = wrapper.offsets_array()
    data = wrapper.data_array()

    return OffsetBlockedArray(offsets, data)
