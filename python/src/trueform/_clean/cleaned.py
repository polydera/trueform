"""
cleaned() function implementation

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Union, Tuple, Optional, Any, Dict
from .. import _trueform
from .._spatial import Mesh, EdgeMesh, PointCloud
from .._core import OffsetBlockedArray

# Dispatch infrastructure
from .._dispatch import InputMeta, build_suffix


def cleaned(
    data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Mesh, EdgeMesh, PointCloud],
    tolerance: Optional[float] = None,
    return_index_map: bool = False
) -> Union[
    np.ndarray,  # just points
    Tuple[np.ndarray, np.ndarray],  # (connectivity, points) or (points, index_map)
    Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
]:
    """
    Remove duplicate vertices and degenerate elements from geometric data.

    Supports points, segment soups, triangle soups, indexed geometry (as tuples),
    meshes (including dynamic), edge meshes, and point clouds. Optionally merges
    vertices within a tolerance distance using spatial trees.

    Parameters
    ----------
    data : np.ndarray, tuple, Mesh, EdgeMesh, or PointCloud
        Input geometric data:
        - Points: shape (N, Dims) where Dims = 2 or 3
        - Segment soup: shape (N, 2, Dims) - each row is a segment with 2 vertices
        - Triangle soup: shape (N, 3, Dims) - each row is a triangle with 3 vertices
        - Indexed geometry: tuple (indices, points) where:
          * indices: shape (N, V) with dtype int32 or int64, V = 2 or 3
            OR OffsetBlockedArray for variable-sized polygons
          * points: shape (M, Dims) where Dims = 2 or 3
        - Mesh: tf.Mesh object (2D or 3D, triangles or dynamic)
        - EdgeMesh: tf.EdgeMesh object (2D or 3D)
        - PointCloud: tf.PointCloud object (2D or 3D)
    tolerance : float, optional
        Distance threshold for merging vertices (default: None = exact duplicates only).
        When specified, a spatial tree is built for efficient proximity queries.
    return_index_map : bool, optional
        If True, return index maps for attribute reindexing (default: False).
        Not supported for polygon/segment soups.

    Returns
    -------
    For points without index map:
        points : np.ndarray
            Cleaned points with shape (M, Dims) where M <= N

    For points with index map:
        (points, point_map) : tuple
            - points: cleaned points (M, Dims)
            - point_map: tuple (f, kept_ids) where:
                * f: int64 array (N,) mapping old id to new id (f[i] == f.size means removed)
                * kept_ids: int64 array of kept old ids

    For meshes/indexed/soups without index map:
        (connectivity, points) : tuple
            - connectivity: cleaned faces/edges with shape (K, V)
            - points: cleaned points with shape (M, Dims)

    For meshes/indexed with index map:
        ((connectivity, points), face_map, point_map) : tuple
            - (connectivity, points): cleaned geometry
            - face_map: tuple (f, kept_ids) for faces/edges, dtype matches input indices
            - point_map: tuple (f, kept_ids) for points, dtype matches input indices

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Clean points (exact duplicates)
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 1, 0]], dtype=np.float32)
    >>> clean_points = tf.cleaned(points)
    >>> print(clean_points.shape)  # (3, 3) - duplicate removed
    >>>
    >>> # Clean points with tolerance and index map
    >>> points = np.array([[0, 0, 0], [0.001, 0, 0], [1, 0, 0]], dtype=np.float32)
    >>> clean_points, (f, kept_ids) = tf.cleaned(points, tolerance=0.01, return_index_map=True)
    >>> # Use f to reindex attributes: new_attrs = old_attrs[kept_ids]
    >>>
    >>> # Clean polygon soup
    >>> soup = np.array([
    ...     [[0, 0, 0], [1, 0, 0], [0.5, 1, 0]],
    ...     [[1, 0, 0], [2, 0, 0], [1.5, 1, 0]]
    ... ], dtype=np.float32)
    >>> faces, points = tf.cleaned(soup)
    >>> print(faces.shape, points.shape)  # (2, 3) (5, 3) - shared vertex deduplicated
    >>>
    >>> # Clean indexed geometry (tuple input)
    >>> indices = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    >>> cleaned_indices, cleaned_points = tf.cleaned((indices, points))
    >>> print(cleaned_indices.shape, cleaned_points.shape)  # (2, 3) (3, 3)
    >>>
    >>> # Clean mesh with index maps for attribute reindexing
    >>> mesh = tf.Mesh(faces, points)
    >>> (clean_faces, clean_points), (f_faces, kept_faces), (f_points, kept_points) = tf.cleaned(
    ...     mesh, return_index_map=True
    ... )
    >>> # Reindex face attributes: new_face_attrs = old_face_attrs[kept_faces]
    >>> # Reindex point attributes: new_point_attrs = old_point_attrs[kept_points]
    """
    # Validate tolerance
    if tolerance is not None and tolerance < 0.0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")

    # Normalize input to (kind, arrays, meta)
    kind, arrays, meta = _extract_cleaned_input(data)

    # Dispatch to appropriate handler
    if kind == 'points':
        return _clean_points(arrays, meta, tolerance, return_index_map)
    elif kind == 'indexed':
        return _clean_indexed(arrays, meta, tolerance, return_index_map)
    elif kind == 'soup':
        if return_index_map:
            raise ValueError("return_index_map is not supported for polygon soups")
        return _clean_soup(arrays, meta, tolerance)


def _extract_cleaned_input(data: Any) -> Tuple[str, tuple, Dict]:
    """
    Normalize any input type to (kind, arrays, meta).

    Returns
    -------
    kind : str
        One of 'points', 'indexed', 'soup'
    arrays : tuple
        Arrays needed for the operation
    meta : dict
        Metadata for suffix building: V, index_dtype, real_dtype, dims, is_dynamic
    """
    # Handle tuple input (indices, points)
    if isinstance(data, tuple):
        return _extract_tuple_input(data)

    # Handle form objects
    if isinstance(data, PointCloud):
        points = data.points
        return ('points', (points,), {
            'real_dtype': points.dtype,
            'dims': data.dims,
        })

    if isinstance(data, (Mesh, EdgeMesh)):
        return _extract_mesh_input(data)

    # Handle numpy arrays
    if isinstance(data, np.ndarray):
        return _extract_array_input(data)

    raise TypeError(
        f"Expected np.ndarray, tuple, or form object (Mesh, EdgeMesh, PointCloud), "
        f"got {type(data).__name__}"
    )


def _extract_tuple_input(data: tuple) -> Tuple[str, tuple, Dict]:
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
        return ('indexed', (indices, points), {
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

        if not indices.flags['C_CONTIGUOUS']:
            indices = np.ascontiguousarray(indices)

        return ('indexed', (indices, points), {
            'V': str(V),
            'index_dtype': indices.dtype,
            'real_dtype': points.dtype,
            'dims': dims,
            'is_dynamic': False,
        })

    raise TypeError(f"indices must be np.ndarray or OffsetBlockedArray, got {type(indices).__name__}")


def _extract_mesh_input(data: Union[Mesh, EdgeMesh]) -> Tuple[str, tuple, Dict]:
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

    return ('indexed', (indices, points), {
        'V': V,
        'index_dtype': indices.dtype,
        'real_dtype': points.dtype,
        'dims': data.dims,
        'is_dynamic': is_dynamic,
    })


def _extract_array_input(data: np.ndarray) -> Tuple[str, tuple, Dict]:
    """Extract and validate numpy array input (points or soup)."""
    if data.dtype not in (np.float32, np.float64):
        raise TypeError(
            f"Data dtype must be float32 or float64, got {data.dtype}. "
            f"Convert with data.astype(np.float32) or data.astype(np.float64)"
        )

    if data.ndim not in (2, 3):
        raise ValueError(
            f"Expected 2D array (points) or 3D array (soup), got {data.ndim}D array with shape {data.shape}"
        )

    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)

    # Points: shape (N, Dims)
    if data.ndim == 2:
        dims = data.shape[1]
        if dims not in (2, 3):
            raise ValueError(f"Points must have 2 or 3 dimensions, got shape {data.shape}")
        return ('points', (data,), {
            'real_dtype': data.dtype,
            'dims': dims,
        })

    # Soup: shape (N, V, Dims)
    V, dims = data.shape[1], data.shape[2]
    if dims not in (2, 3):
        raise ValueError(f"Soup vertices must have 2 or 3 dimensions, got shape {data.shape}")
    if V not in (2, 3):
        raise ValueError(f"Soup elements must have 2 (segments) or 3 (triangles) vertices, got V={V}")

    return ('soup', (data,), {
        'V': V,
        'real_dtype': data.dtype,
        'dims': dims,
    })


def _clean_points(arrays: tuple, meta: Dict, tolerance: Optional[float], return_index_map: bool):
    """Clean points array."""
    points = arrays[0]
    suffix = build_suffix(InputMeta(None, meta['real_dtype'], None, meta['dims']))
    func_name = f"cleaned_points_with_maps_{suffix}"

    cpp_func = getattr(_trueform.clean, func_name)
    result, point_map = cpp_func(points, tolerance)

    if return_index_map:
        return (result, point_map)
    return result


def _clean_indexed(arrays: tuple, meta: Dict, tolerance: Optional[float], return_index_map: bool):
    """Clean indexed geometry (Mesh, EdgeMesh, or tuple)."""
    indices, points = arrays
    suffix = build_suffix(InputMeta(meta['index_dtype'], meta['real_dtype'], meta['V'], meta['dims']))
    func_name = f"cleaned_indexed_with_maps_{suffix}"

    cpp_func = getattr(_trueform.clean, func_name)

    if meta['is_dynamic']:
        # Dynamic mesh - indices is OffsetBlockedArray
        ((offsets, data), result_points), face_map, point_map = cpp_func(
            indices._wrapper, points, tolerance
        )
        result = (OffsetBlockedArray(offsets, data), result_points)
    else:
        result, face_map, point_map = cpp_func(indices, points, tolerance)

    if return_index_map:
        return (result, face_map, point_map)
    return result


def _clean_soup(arrays: tuple, meta: Dict, tolerance: Optional[float]):
    """Clean polygon/segment soup."""
    soup = arrays[0]
    # Soups have no index_dtype, just ngon (V as string), real, dims
    suffix = build_suffix(InputMeta(None, meta['real_dtype'], str(meta['V']), meta['dims']))
    func_name = f"cleaned_soup_{suffix}"

    cpp_func = getattr(_trueform.clean, func_name)
    return cpp_func(soup, tolerance)
