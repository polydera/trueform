"""
reindex_by_ids_on_points() function implementation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Union, Tuple
from .._spatial import Mesh, EdgeMesh
from .._core import OffsetBlockedArray
from .reindex_by_mask_on_points import reindex_by_mask_on_points


def reindex_by_ids_on_points(
    data: Union[Tuple[Union[np.ndarray, OffsetBlockedArray], np.ndarray], Mesh, EdgeMesh],
    point_ids: np.ndarray,
    return_index_map: bool = False
) -> Union[
    Tuple[np.ndarray, np.ndarray],  # (connectivity, points)
    Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],  # ((connectivity, points), face_map, point_map)
]:
    """
    Filter faces/edges based on a set of point IDs.

    Keeps only faces/edges where ALL vertices are in the specified point IDs,
    automatically filtering unused points and maintaining referential integrity.

    Parameters
    ----------
    data : tuple, Mesh, or EdgeMesh
        Input geometric data:
        - Indexed geometry: tuple (indices, points) where:
          * indices: shape (N, V) with dtype int32 or int64, V = 2 or 3,
            OR OffsetBlockedArray for variable-sized polygons
          * points: shape (M, Dims) where Dims = 2 or 3
        - Mesh: tf.Mesh object (2D or 3D, triangles or dynamic)
        - EdgeMesh: tf.EdgeMesh object (2D or 3D)
    point_ids : np.ndarray
        1D array of point IDs to keep, shape (K,) with dtype int32 or int64.
        Faces/edges are kept only if ALL their vertices are in this array.
    return_index_map : bool, optional
        If True, return index maps for attribute reindexing (default: False).

    Returns
    -------
    Without index map:
        (connectivity, points) : tuple
            - connectivity: filtered faces/edges with shape (K, V)
            - points: filtered points with shape (P, Dims)

    With index map:
        ((connectivity, points), face_map, point_map) : tuple
            - (connectivity, points): filtered geometry
            - face_map: tuple (f, kept_ids) for faces/edges, dtype matches input indices
            - point_map: tuple (f, kept_ids) for points, dtype matches input indices

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Filter faces to keep only those using specific vertices
    >>> faces = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]], dtype=np.int32)
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [1.5, 1, 0], [1, 2, 0]], dtype=np.float32)
    >>>
    >>> # Keep only faces using vertices 2, 3, 4
    >>> point_ids = np.array([2, 3, 4], dtype=np.int32)
    >>>
    >>> # Filter - keeps only faces where ALL vertices are in point_ids
    >>> new_faces, new_points = tf.reindex_by_ids_on_points((faces, points), point_ids)
    >>> # Only face [2, 3, 4] survives
    >>>
    >>> # With index maps for attribute reindexing
    >>> (new_faces, new_points), face_map, point_map = tf.reindex_by_ids_on_points(
    ...     (faces, points), point_ids, return_index_map=True
    ... )

    Notes
    -----
    This function filters faces/edges where ALL vertices are in point_ids. If you need
    to keep faces where ANY vertex is in the set, create a mask and use:

        point_mask = np.zeros(n_points, dtype=bool)
        point_mask[point_ids] = True
        face_mask = point_mask[indices].any(axis=1)
        result = tf.reindex_by_mask(data, face_mask)
    """
    # Validate point_ids array
    if not isinstance(point_ids, np.ndarray):
        raise TypeError(
            f"point_ids must be np.ndarray, got {type(point_ids).__name__}"
        )

    if point_ids.ndim != 1:
        raise ValueError(
            f"point_ids must be 1D array with shape (K,), got shape {point_ids.shape}"
        )

    if point_ids.dtype not in (np.int32, np.int64):
        raise TypeError(
            f"point_ids dtype must be int32 or int64, got {point_ids.dtype}. "
            f"Convert with point_ids.astype(np.int32) or point_ids.astype(np.int64)"
        )

    # Get number of points
    if isinstance(data, tuple):
        if len(data) != 2:
            raise ValueError(
                f"Tuple input must have exactly 2 elements (indices, points), got {len(data)}"
            )
        n_points = data[1].shape[0]
    elif isinstance(data, Mesh):
        n_points = data.points.shape[0]
    elif isinstance(data, EdgeMesh):
        n_points = data.points.shape[0]
    else:
        raise TypeError(
            f"Expected tuple or indexed form (Mesh, EdgeMesh), "
            f"got {type(data).__name__}. "
            f"PointCloud is not supported - use reindex_by_ids directly."
        )

    # Convert IDs to mask
    point_mask = np.zeros(n_points, dtype=bool)
    point_mask[point_ids] = True

    return reindex_by_mask_on_points(data, point_mask, return_index_map)
