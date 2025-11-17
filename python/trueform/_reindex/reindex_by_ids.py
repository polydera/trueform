"""
reindex_by_ids() function implementation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

import numpy as np
from typing import Union, Tuple
from .. import _trueform
from .._spatial import Mesh, EdgeMesh, PointCloud


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
          * indices: shape (N, V) with dtype int32 or int64, V = 2, 3, or 4
          * points: shape (M, Dims) where Dims = 2 or 3
        - Mesh: tf.Mesh object (2D or 3D, NGon = 3 or 4)
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
    if not isinstance(ids, np.ndarray):
        raise TypeError(
            f"ids must be np.ndarray, got {type(ids).__name__}"
        )

    if ids.ndim != 1:
        raise ValueError(
            f"ids must be 1D array with shape (N,), got shape {ids.shape}"
        )

    if ids.dtype not in (np.int32, np.int64):
        raise TypeError(
            f"ids dtype must be int32 or int64, got {ids.dtype}. "
            f"Convert with ids.astype(np.int32) or ids.astype(np.int64)"
        )

    # Ensure C-contiguous
    if not ids.flags['C_CONTIGUOUS']:
        ids = np.ascontiguousarray(ids)

    # ===== HANDLE TUPLE INPUT (INDEXED GEOMETRY) =====
    if isinstance(data, tuple):
        if len(data) != 2:
            raise ValueError(
                f"Tuple input must have exactly 2 elements (indices, points), got {len(data)}"
            )

        indices, points = data

        # Validate indices
        if not isinstance(indices, np.ndarray):
            raise TypeError(
                f"indices must be np.ndarray, got {type(indices).__name__}"
            )

        if indices.ndim != 2:
            raise ValueError(
                f"indices must be 2D array with shape (N, V), got shape {indices.shape}"
            )

        if indices.dtype not in (np.int32, np.int64):
            raise TypeError(
                f"indices dtype must be int32 or int64, got {indices.dtype}. "
                f"Convert with indices.astype(np.int32) or indices.astype(np.int64)"
            )

        # Validate points
        if not isinstance(points, np.ndarray):
            raise TypeError(
                f"points must be np.ndarray, got {type(points).__name__}"
            )

        if points.ndim != 2:
            raise ValueError(
                f"points must be 2D array with shape (M, Dims), got shape {points.shape}"
            )

        if points.dtype not in (np.float32, np.float64):
            raise TypeError(
                f"points dtype must be float32 or float64, got {points.dtype}. "
                f"Convert with points.astype(np.float32) or points.astype(np.float64)"
            )

        # Extract shape information
        V = indices.shape[1]
        dims = points.shape[1]

        # Validate V
        if V not in (2, 3, 4):
            raise ValueError(
                f"indices must have 2 (edges), 3 (triangles), or 4 (quads) columns, got V={V}"
            )

        # Validate dims
        if dims not in (2, 3):
            raise ValueError(
                f"points must have 2 or 3 dimensions, got dims={dims}"
            )

        # Ensure indices dtype matches ids dtype
        if indices.dtype != ids.dtype:
            raise TypeError(
                f"indices dtype ({indices.dtype}) must match ids dtype ({ids.dtype})"
            )

        # Ensure C-contiguous
        if not indices.flags['C_CONTIGUOUS']:
            indices = np.ascontiguousarray(indices)
        if not points.flags['C_CONTIGUOUS']:
            points = np.ascontiguousarray(points)

        # Build suffix: {V}{index}{real}{dims}d
        index_str = 'int' if indices.dtype == np.int32 else 'int64'
        real_str = 'float' if points.dtype == np.float32 else 'double'
        suffix = f"{V}{index_str}{real_str}{dims}d"

        # Build function name
        func_name = f"reindexed_by_ids_indexed_{suffix}"

        # Call C++ function - always returns ((connectivity, points), face_map, point_map)
        cpp_func = getattr(_trueform, func_name)
        result, face_map, point_map = cpp_func(indices, points, ids)

        # Conditionally return maps based on user request
        if return_index_map:
            return (result, face_map, point_map)
        else:
            return result

    # ===== HANDLE FORM OBJECTS (Mesh, EdgeMesh, PointCloud) =====
    elif isinstance(data, (Mesh, EdgeMesh, PointCloud)):
        # Validate dims (all support 2D and 3D)
        if data.dims not in (2, 3):
            raise ValueError(
                f"{type(data).__name__} dims must be 2 or 3, got {data.dims}D"
            )

        if isinstance(data, Mesh):
            # Extract arrays from Mesh
            indices = data.faces
            points = data.points
            V = data.ngon

        elif isinstance(data, EdgeMesh):
            # Extract arrays from EdgeMesh
            indices = data.edges
            points = data.points
            V = 2

        else:  # PointCloud
            # PointCloud only has points, no indices
            points = data.points
            dims = data.dims

            # Build suffix: {index}{real}{dims}d
            index_str = 'int' if ids.dtype == np.int32 else 'int64'
            real_str = 'float' if points.dtype == np.float32 else 'double'
            suffix = f"{index_str}{real_str}{dims}d"

            # Build function name
            func_name = f"reindexed_by_ids_points_{suffix}"

            # Call C++ function - always returns (points, point_map)
            cpp_func = getattr(_trueform, func_name)
            result, point_map = cpp_func(points, ids)

            # Conditionally return maps based on user request
            if return_index_map:
                return (result, point_map)
            else:
                return result

        # For Mesh and EdgeMesh, use indexed reindexing
        dims = data.dims

        # Ensure indices dtype matches ids dtype
        if indices.dtype != ids.dtype:
            raise TypeError(
                f"{type(data).__name__} indices dtype ({indices.dtype}) must match ids dtype ({ids.dtype})"
            )

        # Build suffix: {V}{index}{real}{dims}d
        index_str = 'int' if indices.dtype == np.int32 else 'int64'
        real_str = 'float' if points.dtype == np.float32 else 'double'
        suffix = f"{V}{index_str}{real_str}{dims}d"

        # Build function name
        func_name = f"reindexed_by_ids_indexed_{suffix}"

        # Call C++ function - always returns ((connectivity, points), face_map, point_map)
        cpp_func = getattr(_trueform, func_name)
        result, face_map, point_map = cpp_func(indices, points, ids)

        # Conditionally return maps based on user request
        if return_index_map:
            return (result, face_map, point_map)
        else:
            return result

    else:
        raise TypeError(
            f"Expected tuple or form object (Mesh, EdgeMesh, PointCloud), "
            f"got {type(data).__name__}"
        )
