"""
face_link() function implementation

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


def face_link(
    faces: Union[np.ndarray, OffsetBlockedArray],
    cell_membership: OffsetBlockedArray
) -> OffsetBlockedArray:
    """
    Compute face links for faces in a mesh.

    For each face, finds all faces that share at least one edge with it.

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
        Face link structure where block i contains the indices of all faces
        connected to face i by an edge.

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
    ...     [1, 3, 2],
    ...     [2, 3, 4]
    ... ], dtype=np.int32)
    >>> n_ids = 5
    >>>
    >>> membership = tf.cell_membership(faces, n_ids)
    >>> fl = tf.face_link(faces, membership)
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
        func_name = f"compute_face_link_{suffix}"
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
    func_name = f"compute_face_link_{suffix}"
    cpp_func = getattr(_trueform.topology, func_name)

    # Call C++ function - returns offset_blocked_array_wrapper
    wrapper = cpp_func(faces, cell_membership._wrapper)

    return OffsetBlockedArray(wrapper.offsets_array(), wrapper.data_array())
