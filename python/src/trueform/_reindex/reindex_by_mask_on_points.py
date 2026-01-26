"""
reindex_by_mask_on_points() function implementation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Union, Tuple
from .._spatial import Mesh, EdgeMesh
from .._core import OffsetBlockedArray
from .reindex_by_mask import reindex_by_mask


def reindex_by_mask_on_points(
    data: Union[Tuple[Union[np.ndarray, OffsetBlockedArray], np.ndarray], Mesh, EdgeMesh],
    point_mask: np.ndarray,
    return_index_map: bool = False
) -> Union[
    Tuple[np.ndarray, np.ndarray],  # (connectivity, points)
    Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],  # ((connectivity, points), face_map, point_map)
]:
    """
    Filter faces/edges based on a boolean mask over points.

    Keeps only faces/edges where ALL vertices pass the point mask,
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
    point_mask : np.ndarray
        1D boolean array over points, shape (M,) with dtype bool.
        Faces/edges are kept only if ALL their vertices have True in this mask.
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
    >>> # Filter faces to keep only those with vertices above z=0.5
    >>> faces = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]], dtype=np.int32)
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0, 1], [1.5, 0, 1], [1, 0, 2]], dtype=np.float32)
    >>>
    >>> # Create point mask based on z coordinate
    >>> point_mask = points[:, 2] > 0.5  # [False, False, True, True, True]
    >>>
    >>> # Filter - keeps only faces where ALL vertices pass the mask
    >>> new_faces, new_points = tf.reindex_by_mask_on_points((faces, points), point_mask)
    >>> # Only face [2, 3, 4] survives (all vertices have z > 0.5)
    >>>
    >>> # With index maps for attribute reindexing
    >>> (new_faces, new_points), face_map, point_map = tf.reindex_by_mask_on_points(
    ...     (faces, points), point_mask, return_index_map=True
    ... )

    Notes
    -----
    This function filters faces/edges where ALL vertices pass the mask. If you need
    to keep faces where ANY vertex passes, use:

        face_mask = point_mask[indices].any(axis=1)
        result = tf.reindex_by_mask(data, face_mask)
    """
    # Validate point_mask array
    if not isinstance(point_mask, np.ndarray):
        raise TypeError(
            f"point_mask must be np.ndarray, got {type(point_mask).__name__}"
        )

    if point_mask.ndim != 1:
        raise ValueError(
            f"point_mask must be 1D array with shape (M,), got shape {point_mask.shape}"
        )

    if point_mask.dtype != np.bool_:
        raise TypeError(
            f"point_mask dtype must be bool, got {point_mask.dtype}. "
            f"Convert with point_mask.astype(bool)"
        )

    # Extract indices based on data type
    if isinstance(data, tuple):
        if len(data) != 2:
            raise ValueError(
                f"Tuple input must have exactly 2 elements (indices, points), got {len(data)}"
            )
        indices = data[0]
        n_points = data[1].shape[0]
    elif isinstance(data, Mesh):
        indices = data.faces
        n_points = data.points.shape[0]
    elif isinstance(data, EdgeMesh):
        indices = data.edges
        n_points = data.points.shape[0]
    else:
        raise TypeError(
            f"Expected tuple or indexed form (Mesh, EdgeMesh), "
            f"got {type(data).__name__}. "
            f"PointCloud is not supported - use reindex_by_mask directly."
        )

    # Validate point_mask size
    if point_mask.shape[0] != n_points:
        raise ValueError(
            f"point_mask size ({point_mask.shape[0]}) must match number of points ({n_points})"
        )

    # Create face/edge mask: keep only where ALL vertices are True
    if isinstance(indices, OffsetBlockedArray):
        # Dynamic mesh: use reduceat for vectorized per-face reduction
        # minimum.reduceat gives True only if ALL vertices in each face pass
        vertex_passes = point_mask[indices.data]
        face_mask = np.minimum.reduceat(vertex_passes, indices.offsets[:-1])
    else:
        # Fixed-size mesh: vectorized operation
        face_mask = point_mask[indices].all(axis=1)

    return reindex_by_mask(data, face_mask, return_index_map)
