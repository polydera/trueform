"""
reindex_by_ids() function implementation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Union, Tuple, Any, Dict
from .. import _trueform
from .._spatial import Mesh, EdgeMesh, PointCloud
from .._core import OffsetBlockedArray

# Dispatch infrastructure
from .._dispatch import InputMeta, build_suffix


def reindex_by_ids(
    data: Union[Tuple[np.ndarray, np.ndarray], Mesh, EdgeMesh, PointCloud],
    ids: np.ndarray,
    return_index_map: bool = False
) -> Union[
    np.ndarray,  # just points
    Tuple[np.ndarray, np.ndarray],  # (connectivity, points) or (points, index_map)
    Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],  # ((connectivity, points), face_map, point_map)
]:
    """
    Extract specific elements by their IDs from geometric data.

    Reindexes geometry to include only the elements specified by their IDs,
    automatically filtering unused points and maintaining referential integrity.

    Parameters
    ----------
    data : tuple, Mesh, EdgeMesh, or PointCloud
        Input geometric data:
        - Indexed geometry: tuple (indices, points) where:
          * indices: shape (N, V) with dtype int32 or int64, V = 2 or 3
            OR OffsetBlockedArray for variable-sized polygons
          * points: shape (M, Dims) where Dims = 2 or 3
        - Mesh: tf.Mesh object (2D or 3D, triangles or dynamic)
        - EdgeMesh: tf.EdgeMesh object (2D or 3D)
        - PointCloud: tf.PointCloud object (2D or 3D)
    ids : np.ndarray
        1D array of element IDs to extract, shape (K,) with dtype int32 or int64.
        For indexed geometry/Mesh/EdgeMesh: face/edge IDs to keep.
        For PointCloud: point IDs to keep.
    return_index_map : bool, optional
        If True, return index maps for attribute reindexing (default: False).

    Returns
    -------
    For points without index map:
        points : np.ndarray
            Reindexed points with shape (K, Dims)

    For points with index map:
        (points, point_map) : tuple
            - points: reindexed points (K, Dims)
            - point_map: tuple (f, kept_ids) where:
                * f: array (M,) mapping old id to new id, dtype matches ids parameter
                * kept_ids: array of kept old ids, dtype matches ids parameter

    For meshes/indexed without index map:
        (connectivity, points) : tuple
            - connectivity: reindexed faces/edges with shape (K, V)
            - points: reindexed points with shape (P, Dims)

    For meshes/indexed with index map:
        ((connectivity, points), face_map, point_map) : tuple
            - (connectivity, points): reindexed geometry
            - face_map: tuple (f, kept_ids) for faces/edges, dtype matches input indices
            - point_map: tuple (f, kept_ids) for points, dtype matches input indices

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Extract specific faces from mesh
    >>> faces = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]], dtype=np.int32)
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [1.5, 1, 0], [1, 2, 0]], dtype=np.float32)
    >>> face_ids = np.array([0, 2], dtype=np.int32)  # Extract first and third face
    >>>
    >>> # Without index maps
    >>> new_faces, new_points = tf.reindex_by_ids((faces, points), face_ids)
    >>> print(new_faces.shape, new_points.shape)  # (2, 3) (4, 3) - unused point 3 removed
    >>>
    >>> # With index maps for attribute reindexing
    >>> (new_faces, new_points), (f_map, kept_faces), (p_map, kept_points) = tf.reindex_by_ids(
    ...     (faces, points), face_ids, return_index_map=True
    ... )
    >>> # Reindex face attributes: new_face_attrs = old_face_attrs[kept_faces]
    >>> # Reindex point attributes: new_point_attrs = old_point_attrs[kept_points]
    >>>
    >>> # Extract specific points from point cloud
    >>> point_cloud = tf.PointCloud(points)
    >>> point_ids = np.array([0, 2, 4], dtype=np.int32)
    >>> selected_points = tf.reindex_by_ids(point_cloud, point_ids)
    >>> print(selected_points.shape)  # (3, 3)
    """
    # Validate IDs array
    ids = _validate_ids(ids)

    # Normalize input to (kind, arrays, meta)
    kind, arrays, meta = _extract_reindex_input(data, ids)

    # Dispatch to appropriate handler
    if kind == 'points':
        return _reindex_points(arrays, meta, return_index_map)
    else:  # kind == 'indexed'
        return _reindex_indexed(arrays, meta, return_index_map)


def _validate_ids(ids: np.ndarray) -> np.ndarray:
    """Validate and normalize the ids array."""
    if not isinstance(ids, np.ndarray):
        raise TypeError(f"ids must be np.ndarray, got {type(ids).__name__}")

    if ids.ndim != 1:
        raise ValueError(f"ids must be 1D array with shape (N,), got shape {ids.shape}")

    if ids.dtype not in (np.int32, np.int64):
        raise TypeError(
            f"ids dtype must be int32 or int64, got {ids.dtype}. "
            f"Convert with ids.astype(np.int32) or ids.astype(np.int64)"
        )

    if not ids.flags['C_CONTIGUOUS']:
        ids = np.ascontiguousarray(ids)

    return ids


def _extract_reindex_input(data: Any, ids: np.ndarray) -> Tuple[str, tuple, Dict]:
    """
    Normalize any input type to (kind, arrays, meta).

    Returns
    -------
    kind : str
        One of 'points', 'indexed'
    arrays : tuple
        Arrays needed for the operation
    meta : dict
        Metadata for suffix building and processing
    """
    # Handle tuple input (indices, points)
    if isinstance(data, tuple):
        return _extract_tuple_input(data, ids)

    # Handle form objects
    if isinstance(data, PointCloud):
        points = data.points
        return ('points', (points, ids), {
            'index_dtype': ids.dtype,
            'real_dtype': points.dtype,
            'dims': data.dims,
        })

    if isinstance(data, (Mesh, EdgeMesh)):
        return _extract_mesh_input(data, ids)

    raise TypeError(
        f"Expected tuple or form object (Mesh, EdgeMesh, PointCloud), "
        f"got {type(data).__name__}"
    )


def _extract_tuple_input(data: tuple, ids: np.ndarray) -> Tuple[str, tuple, Dict]:
    """Extract and validate (indices, points) tuple input."""
    if len(data) != 2:
        raise ValueError(
            f"Tuple input must have exactly 2 elements (indices, points), got {len(data)}"
        )

    indices, points = data

    # Validate points
    if not isinstance(points, np.ndarray):
        raise TypeError(f"points must be np.ndarray, got {type(points).__name__}")
    if points.ndim != 2:
        raise ValueError(f"points must be 2D array with shape (M, Dims), got shape {points.shape}")
    if points.dtype not in (np.float32, np.float64):
        raise TypeError(
            f"points dtype must be float32 or float64, got {points.dtype}. "
            f"Convert with points.astype(np.float32) or points.astype(np.float64)"
        )

    dims = points.shape[1]
    if dims not in (2, 3):
        raise ValueError(f"points must have 2 or 3 dimensions, got dims={dims}")

    # Ensure C-contiguous
    if not points.flags['C_CONTIGUOUS']:
        points = np.ascontiguousarray(points)

    # Handle dynamic (OffsetBlockedArray) indices
    if isinstance(indices, OffsetBlockedArray):
        if indices.dtype not in (np.int32, np.int64):
            raise TypeError(f"indices dtype must be int32 or int64, got {indices.dtype}")
        if indices.dtype != ids.dtype:
            raise TypeError(
                f"indices dtype ({indices.dtype}) must match ids dtype ({ids.dtype})"
            )
        return ('indexed', (indices, points, ids), {
            'V': 'dyn',
            'index_dtype': indices.dtype,
            'real_dtype': points.dtype,
            'dims': dims,
            'is_dynamic': True,
        })

    # Handle fixed-size (ndarray) indices
    if isinstance(indices, np.ndarray):
        if indices.ndim != 2:
            raise ValueError(f"indices must be 2D array with shape (N, V), got shape {indices.shape}")
        if indices.dtype not in (np.int32, np.int64):
            raise TypeError(
                f"indices dtype must be int32 or int64, got {indices.dtype}. "
                f"Convert with indices.astype(np.int32) or indices.astype(np.int64)"
            )

        V = indices.shape[1]
        if V not in (2, 3):
            raise ValueError(
                f"Fixed-size indices must have 2 (edges) or 3 (triangles) columns, got V={V}. "
                f"For variable-sized polygons, use OffsetBlockedArray."
            )

        if indices.dtype != ids.dtype:
            raise TypeError(
                f"indices dtype ({indices.dtype}) must match ids dtype ({ids.dtype})"
            )

        if not indices.flags['C_CONTIGUOUS']:
            indices = np.ascontiguousarray(indices)

        return ('indexed', (indices, points, ids), {
            'V': str(V),
            'index_dtype': indices.dtype,
            'real_dtype': points.dtype,
            'dims': dims,
            'is_dynamic': False,
        })

    raise TypeError(f"indices must be np.ndarray or OffsetBlockedArray, got {type(indices).__name__}")


def _extract_mesh_input(data: Union[Mesh, EdgeMesh], ids: np.ndarray) -> Tuple[str, tuple, Dict]:
    """Extract and validate Mesh/EdgeMesh input."""
    if data.dims not in (2, 3):
        raise ValueError(f"{type(data).__name__} dims must be 2 or 3, got {data.dims}D")

    if isinstance(data, Mesh):
        indices = data.faces
        points = data.points
        is_dynamic = data.is_dynamic
        V = 'dyn' if is_dynamic else str(data.ngon)
    else:  # EdgeMesh
        indices = data.edges
        points = data.points
        is_dynamic = False
        V = '2'

    # Verify dtype match
    if indices.dtype != ids.dtype:
        raise TypeError(
            f"{type(data).__name__} indices dtype ({indices.dtype}) must match ids dtype ({ids.dtype})"
        )

    return ('indexed', (indices, points, ids), {
        'V': V,
        'index_dtype': indices.dtype,
        'real_dtype': points.dtype,
        'dims': data.dims,
        'is_dynamic': is_dynamic,
    })


def _reindex_points(arrays: tuple, meta: Dict, return_index_map: bool):
    """Reindex points (PointCloud case)."""
    points, ids = arrays
    # Points with ids: {index}{real}{dims}d (no ngon)
    suffix = build_suffix(InputMeta(meta['index_dtype'], meta['real_dtype'], None, meta['dims']))
    func_name = f"reindexed_by_ids_points_{suffix}"

    cpp_func = getattr(_trueform.reindex, func_name)
    result, point_map = cpp_func(points, ids)

    if return_index_map:
        return (result, point_map)
    return result


def _reindex_indexed(arrays: tuple, meta: Dict, return_index_map: bool):
    """Reindex indexed geometry (Mesh, EdgeMesh, or tuple)."""
    indices, points, ids = arrays
    suffix = build_suffix(InputMeta(meta['index_dtype'], meta['real_dtype'], meta['V'], meta['dims']))
    func_name = f"reindexed_by_ids_indexed_{suffix}"

    cpp_func = getattr(_trueform.reindex, func_name)

    if meta['is_dynamic']:
        # Dynamic mesh - indices is OffsetBlockedArray
        ((offsets, data_arr), result_points), face_map, point_map = cpp_func(
            indices._wrapper, points, ids
        )
        result = (OffsetBlockedArray(offsets, data_arr), result_points)
    else:
        result, face_map, point_map = cpp_func(indices, points, ids)

    if return_index_map:
        return (result, face_map, point_map)
    return result
