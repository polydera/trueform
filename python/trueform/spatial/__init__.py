"""
Spatial query operations

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

import numpy as np
from typing import Any, Optional, Union, List, Tuple
from .. import _trueform
from ..primitives import Point, Segment, Polygon, Ray, Line
from ._point_cloud_neighbor_search import (
    _POINT_CLOUD_NEIGHBOR_SEARCH_DISPATCH,
    _POINT_CLOUD_NEIGHBOR_SEARCH_KNN_DISPATCH
)

# Unified dispatch tables
# Currently only point cloud dispatch, will merge with mesh dispatch later
_NEIGHBOR_SEARCH_DISPATCH = {**_POINT_CLOUD_NEIGHBOR_SEARCH_DISPATCH}
# When mesh module is added:
# from .spatial._mesh_neighbor_search import _MESH_NEIGHBOR_SEARCH_DISPATCH
# _NEIGHBOR_SEARCH_DISPATCH = {**_POINT_CLOUD_NEIGHBOR_SEARCH_DISPATCH, **_MESH_NEIGHBOR_SEARCH_DISPATCH}

_NEIGHBOR_SEARCH_KNN_DISPATCH = {**_POINT_CLOUD_NEIGHBOR_SEARCH_KNN_DISPATCH}
# When mesh module is added:
# from .spatial._mesh_neighbor_search import _MESH_NEIGHBOR_SEARCH_KNN_DISPATCH
# _NEIGHBOR_SEARCH_KNN_DISPATCH = {**_POINT_CLOUD_NEIGHBOR_SEARCH_KNN_DISPATCH, **_MESH_NEIGHBOR_SEARCH_KNN_DISPATCH}


def neighbor_search(
    point_cloud: Any,
    query: Any,
    radius: Optional[float] = None,
    k: Optional[int] = None
) -> Union[Tuple[int, float, np.ndarray], List[Tuple[int, float, np.ndarray]]]:
    """
    Search for nearest neighbor(s) in a point cloud.

    Performs spatial queries to find the closest point(s) in a point cloud to a given
    geometric primitive (point, segment, polygon, ray, or line).

    Parameters
    ----------
    point_cloud : PointCloud
        The point cloud to search in
    query : Point, Segment, Polygon, Ray, Line, or numpy array
        The geometric primitive to query with. Can be a wrapped primitive or a numpy array
        (which will be treated as a Point)
    radius : float, optional
        Maximum search radius. If None, searches without distance limit.
        For non-KNN queries, only returns a result if within this radius.
        For KNN queries, limits results to neighbors within this radius.
    k : int, optional
        Number of nearest neighbors to find. If None, returns only the single nearest neighbor.
        If specified, returns up to k nearest neighbors (may be fewer if limited by radius).

    Returns
    -------
    single_result : tuple[int, float, ndarray]
        When k is None: Returns (index, distance_squared, point) for the nearest neighbor,
        where index is the point index in the cloud, distance_squared is the squared distance,
        and point is the coordinates of the closest point on the query primitive.
    multiple_results : list[tuple[int, float, ndarray]]
        When k is specified: Returns a list of up to k tuples (index, distance_squared, point)
        sorted by distance (closest first).

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>> # Create a 3D point cloud
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    >>> cloud = tf.PointCloud(points)
    >>>
    >>> # Find nearest neighbor to a point
    >>> query_pt = tf.Point([0.1, 0.1, 0.0])
    >>> idx, dist2, closest_pt = tf.neighbor_search(cloud, query_pt)
    >>> print(f"Nearest point index: {idx}, distance²: {dist2}")
    >>>
    >>> # Find 2 nearest neighbors within radius 2.0
    >>> results = tf.neighbor_search(cloud, query_pt, radius=2.0, k=2)
    >>> for idx, dist2, pt in results:
    ...     print(f"Index {idx}: distance²={dist2}, point={pt}")
    >>>
    >>> # Query with a segment
    >>> seg = tf.Segment([[0.5, 0.5, 0], [0.5, 0.5, 1]])
    >>> idx, dist2, closest_pt = tf.neighbor_search(cloud, seg)
    """

    # Normalize query to a primitive type
    if isinstance(query, np.ndarray):
        # Treat numpy arrays as points
        if query.ndim == 1:
            # Infer dimensions from array shape
            dims = query.shape[0]
            query_type = Point
            query_data = query
        else:
            raise TypeError(
                f"numpy array queries must be 1D point arrays, got shape {query.shape}"
            )
    else:
        query_type = type(query)
        query_data = query.data if hasattr(query, 'data') else query
        dims = query.dims if hasattr(query, 'dims') else None

    # Validate dimensions match
    cloud_dims = point_cloud.dims
    if dims is None:
        raise TypeError(f"Cannot determine dimensions for query type {query_type}")
    if cloud_dims != dims:
        raise ValueError(
            f"Dimension mismatch: point_cloud has {cloud_dims}D, query has {dims}D. "
            f"Both must have the same dimensionality (2D or 3D)."
        )

    # Get variant suffix - MUST use point cloud's dtype since wrapper is already typed
    # The C++ functions expect all types (cloud wrapper, query array, radius) to match
    cloud_dtype = point_cloud.points.dtype
    dtype_str = 'float' if cloud_dtype == np.float32 else 'double'
    suffix = f"{dtype_str}{cloud_dims}d"

    # Convert query_data to match cloud dtype if necessary
    if isinstance(query_data, np.ndarray) and query_data.dtype != cloud_dtype:
        query_data = query_data.astype(cloud_dtype)

    # Choose dispatch table based on whether k is specified
    if k is None:
        # Non-KNN query - single nearest neighbor
        if query_type not in _NEIGHBOR_SEARCH_DISPATCH:
            supported = ", ".join(t.__name__ for t in _NEIGHBOR_SEARCH_DISPATCH.keys())
            raise TypeError(
                f"neighbor_search not implemented for query type: {query_type.__name__}. "
                f"Supported types: {supported}"
            )

        func_name = _NEIGHBOR_SEARCH_DISPATCH[query_type].format(suffix)
        cpp_func = getattr(_trueform.spatial, func_name)
        return cpp_func(point_cloud._wrapper, query_data, radius)
    else:
        # KNN query - k nearest neighbors
        if query_type not in _NEIGHBOR_SEARCH_KNN_DISPATCH:
            supported = ", ".join(t.__name__ for t in _NEIGHBOR_SEARCH_KNN_DISPATCH.keys())
            raise TypeError(
                f"neighbor_search (KNN) not implemented for query type: {query_type.__name__}. "
                f"Supported types: {supported}"
            )

        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")

        func_name = _NEIGHBOR_SEARCH_KNN_DISPATCH[query_type].format(suffix)
        cpp_func = getattr(_trueform.spatial, func_name)
        return cpp_func(point_cloud._wrapper, query_data, k, radius)


__all__ = ['neighbor_search']
