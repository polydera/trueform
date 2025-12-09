"""
reindex_by_mask() function implementation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

import numpy as np
from typing import Union, Tuple
from .. import _trueform
from .._spatial import Mesh, EdgeMesh, PointCloud
from .._core import OffsetBlockedArray


def reindex_by_mask(
    data: Union[Tuple[np.ndarray, np.ndarray], Mesh, EdgeMesh, PointCloud],
    mask: np.ndarray,
    return_index_map: bool = False
) -> Union[
    np.ndarray,  # just points
    Tuple[np.ndarray, np.ndarray],  # (connectivity, points) or (points, index_map)
    Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],  # ((connectivity, points), face_map, point_map)
]:
    """
    Filter geometric data using a boolean mask.

    Reindexes geometry to include only elements where the mask is True,
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
    mask : np.ndarray
        1D boolean array indicating which elements to keep, shape (N,) with dtype bool.
        For indexed geometry/Mesh/EdgeMesh: mask over faces/edges.
        For PointCloud: mask over points.
    return_index_map : bool, optional
        If True, return index maps for attribute reindexing (default: False).

    Returns
    -------
    For points without index map:
        points : np.ndarray
            Filtered points with shape (K, Dims) where K = mask.sum()

    For points with index map:
        (points, point_map) : tuple
            - points: filtered points (K, Dims)
            - point_map: tuple (f, kept_ids) where:
                * f: int64 array (M,) mapping old id to new id
                * kept_ids: int64 array of kept old ids

    For meshes/indexed without index map:
        (connectivity, points) : tuple
            - connectivity: filtered faces/edges with shape (K, V)
            - points: filtered points with shape (P, Dims)

    For meshes/indexed with index map:
        ((connectivity, points), face_map, point_map) : tuple
            - (connectivity, points): filtered geometry
            - face_map: tuple (f, kept_ids) for faces/edges, dtype matches input indices
            - point_map: tuple (f, kept_ids) for points, dtype matches input indices

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Filter faces by area threshold
    >>> faces = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]], dtype=np.int32)
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [1.5, 1, 0], [1, 2, 0]], dtype=np.float32)
    >>>
    >>> # Create mask (e.g., based on some criterion)
    >>> face_mask = np.array([True, False, True], dtype=bool)  # Keep first and third face
    >>>
    >>> # Without index maps
    >>> new_faces, new_points = tf.reindex_by_mask((faces, points), face_mask)
    >>> print(new_faces.shape, new_points.shape)  # (2, 3) (4, 3) - unused point 3 removed
    >>>
    >>> # With index maps for attribute reindexing
    >>> (new_faces, new_points), (f_map, kept_faces), (p_map, kept_points) = tf.reindex_by_mask(
    ...     (faces, points), face_mask, return_index_map=True
    ... )
    >>> # Reindex face attributes: new_face_attrs = old_face_attrs[kept_faces]
    >>> # Reindex point attributes: new_point_attrs = old_point_attrs[kept_points]
    >>>
    >>> # Filter points from point cloud
    >>> point_cloud = tf.PointCloud(points)
    >>> point_mask = np.array([True, False, True, False, True], dtype=bool)
    >>> filtered_points = tf.reindex_by_mask(point_cloud, point_mask)
    >>> print(filtered_points.shape)  # (3, 3)
    """

    # Validate mask array
    if not isinstance(mask, np.ndarray):
        raise TypeError(
            f"mask must be np.ndarray, got {type(mask).__name__}"
        )

    if mask.ndim != 1:
        raise ValueError(
            f"mask must be 1D array with shape (N,), got shape {mask.shape}"
        )

    if mask.dtype != np.bool_:
        raise TypeError(
            f"mask dtype must be bool, got {mask.dtype}. "
            f"Convert with mask.astype(bool)"
        )

    # Ensure C-contiguous
    if not mask.flags['C_CONTIGUOUS']:
        mask = np.ascontiguousarray(mask)

    # ===== HANDLE TUPLE INPUT (INDEXED GEOMETRY) =====
    if isinstance(data, tuple):
        if len(data) != 2:
            raise ValueError(
                f"Tuple input must have exactly 2 elements (indices, points), got {len(data)}"
            )

        indices, points = data

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

        dims = points.shape[1]

        # Validate dims
        if dims not in (2, 3):
            raise ValueError(
                f"points must have 2 or 3 dimensions, got dims={dims}"
            )

        # Ensure C-contiguous for points
        if not points.flags['C_CONTIGUOUS']:
            points = np.ascontiguousarray(points)

        # Handle dynamic (OffsetBlockedArray) indices
        if isinstance(indices, OffsetBlockedArray):
            if indices.dtype not in (np.int32, np.int64):
                raise TypeError(
                    f"indices dtype must be int32 or int64, got {indices.dtype}"
                )

            # Validate mask size
            if mask.shape[0] != len(indices):
                raise ValueError(
                    f"mask size ({mask.shape[0]}) must match number of faces ({len(indices)})"
                )

            # Build suffix: dyn{index}{real}{dims}d
            index_str = 'int' if indices.dtype == np.int32 else 'int64'
            real_str = 'float' if points.dtype == np.float32 else 'double'
            suffix = f"dyn{index_str}{real_str}{dims}d"

            # Build function name
            func_name = f"reindexed_by_mask_indexed_{suffix}"

            # Call C++ function
            # Returns: ((offsets, data), points), face_map, point_map
            cpp_func = getattr(_trueform.reindex, func_name)
            ((offsets, data_arr), result_points), face_map, point_map = cpp_func(
                indices._wrapper, points, mask
            )

            # Convert to OffsetBlockedArray
            result = (OffsetBlockedArray(offsets, data_arr), result_points)

            if return_index_map:
                return (result, face_map, point_map)
            else:
                return result

        # Handle fixed-size (ndarray) indices
        elif isinstance(indices, np.ndarray):
            if indices.ndim != 2:
                raise ValueError(
                    f"indices must be 2D array with shape (N, V), got shape {indices.shape}"
                )

            if indices.dtype not in (np.int32, np.int64):
                raise TypeError(
                    f"indices dtype must be int32 or int64, got {indices.dtype}. "
                    f"Convert with indices.astype(np.int32) or indices.astype(np.int64)"
                )

            # Extract shape information
            V = indices.shape[1]

            # Validate V (only triangles and edges for fixed-size)
            if V not in (2, 3):
                raise ValueError(
                    f"Fixed-size indices must have 2 (edges) or 3 (triangles) columns, got V={V}. "
                    f"For variable-sized polygons, use OffsetBlockedArray."
                )

            # Validate mask size
            if mask.shape[0] != indices.shape[0]:
                raise ValueError(
                    f"mask size ({mask.shape[0]}) must match number of faces/edges ({indices.shape[0]})"
                )

            # Ensure C-contiguous
            if not indices.flags['C_CONTIGUOUS']:
                indices = np.ascontiguousarray(indices)

            # Build suffix: {V}{index}{real}{dims}d
            index_str = 'int' if indices.dtype == np.int32 else 'int64'
            real_str = 'float' if points.dtype == np.float32 else 'double'
            suffix = f"{V}{index_str}{real_str}{dims}d"

            # Build function name
            func_name = f"reindexed_by_mask_indexed_{suffix}"

            # Call C++ function - always returns ((connectivity, points), face_map, point_map)
            cpp_func = getattr(_trueform.reindex, func_name)
            result, face_map, point_map = cpp_func(indices, points, mask)

            # Conditionally return maps based on user request
            if return_index_map:
                return (result, face_map, point_map)
            else:
                return result

        else:
            raise TypeError(
                f"indices must be np.ndarray or OffsetBlockedArray, got {type(indices).__name__}"
            )

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
            is_dynamic = data.is_dynamic
            V = None if is_dynamic else data.ngon

            # Validate mask size
            if mask.shape[0] != data.number_of_faces:
                raise ValueError(
                    f"mask size ({mask.shape[0]}) must match number of faces ({data.number_of_faces})"
                )

        elif isinstance(data, EdgeMesh):
            # Extract arrays from EdgeMesh
            indices = data.edges
            points = data.points
            is_dynamic = False
            V = 2

            # Validate mask size
            if mask.shape[0] != indices.shape[0]:
                raise ValueError(
                    f"mask size ({mask.shape[0]}) must match number of edges ({indices.shape[0]})"
                )

        else:  # PointCloud
            # PointCloud only has points, no indices
            points = data.points
            dims = data.dims

            # Validate mask size
            if mask.shape[0] != points.shape[0]:
                raise ValueError(
                    f"mask size ({mask.shape[0]}) must match number of points ({points.shape[0]})"
                )

            # Build suffix: {real}{dims}d
            real_str = 'float' if points.dtype == np.float32 else 'double'
            suffix = f"{real_str}{dims}d"

            # Build function name
            func_name = f"reindexed_by_mask_points_{suffix}"

            # Call C++ function - always returns (points, point_map)
            cpp_func = getattr(_trueform.reindex, func_name)
            result, point_map = cpp_func(points, mask)

            # Conditionally return maps based on user request
            if return_index_map:
                return (result, point_map)
            else:
                return result

        # For Mesh and EdgeMesh, use indexed reindexing
        dims = data.dims
        index_str = 'int' if indices.dtype == np.int32 else 'int64'
        real_str = 'float' if points.dtype == np.float32 else 'double'

        if is_dynamic:
            # Dynamic mesh - indices is OffsetBlockedArray
            suffix = f"dyn{index_str}{real_str}{dims}d"
            func_name = f"reindexed_by_mask_indexed_{suffix}"

            cpp_func = getattr(_trueform.reindex, func_name)
            # Returns: ((offsets, data), points), face_map, point_map
            ((offsets, data_arr), result_points), face_map, point_map = cpp_func(
                indices._wrapper, points, mask
            )

            # Convert to OffsetBlockedArray
            result = (OffsetBlockedArray(offsets, data_arr), result_points)
        else:
            # Fixed-size mesh/edge mesh
            suffix = f"{V}{index_str}{real_str}{dims}d"
            func_name = f"reindexed_by_mask_indexed_{suffix}"

            cpp_func = getattr(_trueform.reindex, func_name)
            result, face_map, point_map = cpp_func(indices, points, mask)

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
