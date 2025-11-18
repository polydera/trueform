"""
face_link() function implementation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

import numpy as np
from .. import _trueform
from .._core import OffsetBlockedArray


def face_link(
    faces: np.ndarray,
    cell_membership: OffsetBlockedArray
) -> OffsetBlockedArray:
    """
    Compute face links for vertices in a mesh.

    For each vertex, finds all faces that share at least one edge with any face
    containing that vertex (i.e., faces in the vertex's link).

    Parameters
    ----------
    faces : np.ndarray
        Face connectivity array of shape (N, V) where N is number of faces and
        V is vertices per face (3 for triangles, 4 for quads).
        Must have dtype int32 or int64.

    cell_membership : OffsetBlockedArray
        Cell membership structure mapping vertices to faces. Can be computed
        using cell_membership().

    Returns
    -------
    OffsetBlockedArray
        Face link structure where block i contains the indices of all faces
        in the link of vertex i.

    Raises
    ------
    TypeError
        If faces is not np.ndarray, has wrong dtype, or cell_membership is not
        OffsetBlockedArray
    ValueError
        If faces has wrong shape or V is not 3 or 4

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

    # ===== VALIDATE faces =====
    if not isinstance(faces, np.ndarray):
        raise TypeError(
            f"faces must be np.ndarray, got {type(faces).__name__}"
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

    # Validate ngon (V) - only triangles and quads
    ngon = faces.shape[1]
    if ngon not in (3, 4):
        raise ValueError(
            f"faces must have 3 or 4 vertices per face, got {ngon}. "
            f"Face link is only defined for triangles and quads."
        )

    # ===== VALIDATE cell_membership =====
    if not isinstance(cell_membership, OffsetBlockedArray):
        raise TypeError(
            f"cell_membership must be OffsetBlockedArray, "
            f"got {type(cell_membership).__name__}"
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
    dtype_str = 'int' if faces.dtype == np.int32 else 'int64'
    suffix = f"{dtype_str}_{ngon}"

    func_name = f"compute_face_link_{suffix}"
    cpp_func = getattr(_trueform.topology, func_name)

    # Call C++ function - returns tuple of (offsets, data)
    offsets, data = cpp_func(faces, cell_membership._wrapper)

    return OffsetBlockedArray(offsets, data)
