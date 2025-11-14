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
from .._primitives import Point, Segment, Polygon, Ray, Line
from .._core.mesh import Mesh
from .._core.edge_mesh import EdgeMesh
from .._core.point_cloud import PointCloud
from ._point_cloud_neighbor_search import (
    _POINT_CLOUD_NEIGHBOR_SEARCH_DISPATCH,
    _POINT_CLOUD_NEIGHBOR_SEARCH_KNN_DISPATCH
)
from ._mesh_neighbor_search import (
    _MESH_NEIGHBOR_SEARCH_DISPATCH,
    _MESH_NEIGHBOR_SEARCH_KNN_DISPATCH
)
from ._edge_mesh_neighbor_search import (
    _EDGE_MESH_NEIGHBOR_SEARCH_DISPATCH,
    _EDGE_MESH_NEIGHBOR_SEARCH_KNN_DISPATCH
)
from ._form_form_neighbor_search import _FORM_FORM_NEIGHBOR_SEARCH_DISPATCH
from .gather_ids import gather_intersecting_ids, gather_ids_within_distance

# Dispatch tables organized by object type
_DISPATCH_BY_TYPE = {
    'PointCloud': {
        'single': _POINT_CLOUD_NEIGHBOR_SEARCH_DISPATCH,
        'knn': _POINT_CLOUD_NEIGHBOR_SEARCH_KNN_DISPATCH
    },
    'Mesh': {
        'single': _MESH_NEIGHBOR_SEARCH_DISPATCH,
        'knn': _MESH_NEIGHBOR_SEARCH_KNN_DISPATCH
    },
    'EdgeMesh': {
        'single': _EDGE_MESH_NEIGHBOR_SEARCH_DISPATCH,
        'knn': _EDGE_MESH_NEIGHBOR_SEARCH_KNN_DISPATCH
    }
}


def neighbor_search(
    spatial_object: Any,
    query: Any,
    radius: Optional[float] = None,
    k: Optional[int] = None
) -> Union[Tuple[int, float, np.ndarray], List[Tuple[int, float, np.ndarray]], Tuple[Tuple[int, int], Tuple[float, np.ndarray, np.ndarray]]]:
    """
    Search for nearest neighbor(s) in a spatial structure.

    Performs spatial queries to find the closest element(s) in a point cloud, mesh, or edge mesh to a given
    geometric primitive (point, segment, polygon, ray, or line) or another spatial form.

    Parameters
    ----------
    spatial_object : PointCloud, Mesh, or EdgeMesh
        The spatial structure to search in
    query : Point, Segment, Polygon, Ray, Line, PointCloud, Mesh, EdgeMesh, or numpy array
        The geometric primitive or spatial form to query with. Can be a wrapped primitive,
        another spatial form, or a numpy array (which will be treated as a Point)
    radius : float, optional
        Maximum search radius. If None, searches without distance limit.
        For non-KNN queries, only returns a result if within this radius.
        For KNN queries, limits results to neighbors within this radius.
    k : int, optional
        Number of nearest neighbors to find. If None, returns only the single nearest neighbor.
        If specified, returns up to k nearest neighbors (may be fewer if limited by radius).
        Note: KNN is not supported for form-form queries.

    Returns
    -------
    single_result : tuple[int, float, ndarray] (form-primitive query)
        When k is None: Returns (index, distance_squared, point) for the nearest neighbor,
        where index is the element index, distance_squared is the squared distance,
        and point is the coordinates of the closest point on the query primitive.
    form_form_result : tuple[tuple[int, int], tuple[float, ndarray, ndarray]] (form-form query)
        When query is a form: Returns ((index0, index1), (distance, point0, point1)) where
        index0 and index1 are element indices in the two forms, distance is the squared distance,
        and point0, point1 are the closest points on each form.
    multiple_results : list[tuple[int, float, ndarray]] (form-primitive KNN)
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
    >>> # Query a mesh with a segment
    >>> faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    >>> mesh = tf.Mesh(faces, points)
    >>> seg = tf.Segment([[0.5, 0.5, 0], [0.5, 0.5, 1]])
    >>> idx, dist2, closest_pt = tf.neighbor_search(mesh, seg)
    >>>
    >>> # Form-form neighbor search
    >>> cloud2 = tf.PointCloud(points + 0.5)
    >>> (idx0, idx1), (dist, pt0, pt1) = tf.neighbor_search(cloud, cloud2)
    >>> print(f"Closest pair: cloud[{idx0}] to cloud2[{idx1}], distance²={dist}")
    """

    # Check if query is also a form (for form-form neighbor search)
    query_type = type(query)
    is_form_form = query_type in (Mesh, EdgeMesh, PointCloud)

    if is_form_form:
        # Form-Form neighbor search
        if k is not None:
            raise ValueError("KNN (k parameter) is not supported for form-form neighbor_search")

        # Validate dimensions match
        if spatial_object.dims != query.dims:
            raise ValueError(
                f"Dimension mismatch: first form has {spatial_object.dims}D, "
                f"second form has {query.dims}D. Both must have the same dimensionality (2D or 3D)."
            )

        # Get type pair
        type_pair = (type(spatial_object), query_type)

        if type_pair not in _FORM_FORM_NEIGHBOR_SEARCH_DISPATCH:
            supported = set()
            for t0, t1 in _FORM_FORM_NEIGHBOR_SEARCH_DISPATCH.keys():
                supported.add(t0.__name__)
                supported.add(t1.__name__)
            raise TypeError(
                f"form-form neighbor_search not implemented for types: {type(spatial_object).__name__}, {query_type.__name__}. "
                f"Supported types: {', '.join(sorted(supported))}"
            )

        func_template, needs_swap = _FORM_FORM_NEIGHBOR_SEARCH_DISPATCH[type_pair]

        # Determine the canonical order
        form0_obj = spatial_object if not needs_swap else query
        form1_obj = query if not needs_swap else spatial_object
        form0_type = type(form0_obj)
        form1_type = type(form1_obj)

        # Canonicalize index type ordering for same-type forms
        # C++ only implements: int×int, int×int64, int64×int64
        # If we have int64×int, swap to int×int64
        extra_swap = False
        if form0_type == form1_type:
            if form0_type is EdgeMesh:
                index0_dtype = form0_obj.edges.dtype
                index1_dtype = form1_obj.edges.dtype
                # Swap if form0 is int64 and form1 is int32
                if index0_dtype == np.int64 and index1_dtype == np.int32:
                    form0_obj, form1_obj = form1_obj, form0_obj
                    extra_swap = True
            elif form0_type is Mesh:
                index0_dtype = form0_obj.faces.dtype
                index1_dtype = form1_obj.faces.dtype
                # Swap if form0 is int64 and form1 is int32
                if index0_dtype == np.int64 and index1_dtype == np.int32:
                    form0_obj, form1_obj = form1_obj, form0_obj
                    extra_swap = True

        # Get real type (must match)
        if form0_type is PointCloud:
            real_str = 'float' if form0_obj.points.dtype == np.float32 else 'double'
        elif form0_type is Mesh:
            real_str = 'float' if form0_obj.points.dtype == np.float32 else 'double'
        else:  # EdgeMesh
            real_str = 'float' if form0_obj.points.dtype == np.float32 else 'double'

        # Dims (must match)
        dims_str = f"{form0_obj.dims}d"

        # Build suffix based on form types
        if form0_type is PointCloud and form1_type is PointCloud:
            suffix = f"{real_str}{dims_str}"
        elif form0_type is EdgeMesh and form1_type is EdgeMesh:
            index0_str = 'int' if form0_obj.edges.dtype == np.int32 else 'int64'
            index1_str = 'int' if form1_obj.edges.dtype == np.int32 else 'int64'
            suffix = f"{index0_str}{index1_str}{real_str}{dims_str}"
        elif form0_type is EdgeMesh and form1_type is PointCloud:
            index0_str = 'int' if form0_obj.edges.dtype == np.int32 else 'int64'
            suffix = f"{index0_str}{real_str}{dims_str}"
        elif form0_type is Mesh and form1_type is PointCloud:
            index0_str = 'int' if form0_obj.faces.dtype == np.int32 else 'int64'
            suffix = f"{index0_str}{real_str}{form0_obj.ngon}{dims_str}"
        elif form0_type is Mesh and form1_type is EdgeMesh:
            index0_str = 'int' if form0_obj.faces.dtype == np.int32 else 'int64'
            index1_str = 'int' if form1_obj.edges.dtype == np.int32 else 'int64'
            suffix = f"{index0_str}{index1_str}{real_str}{form0_obj.ngon}{dims_str}"
        elif form0_type is Mesh and form1_type is Mesh:
            index0_str = 'int' if form0_obj.faces.dtype == np.int32 else 'int64'
            index1_str = 'int' if form1_obj.faces.dtype == np.int32 else 'int64'
            suffix = f"{index0_str}{index1_str}{form0_obj.ngon}{form1_obj.ngon}{real_str}{dims_str}"
        else:
            raise TypeError(f"Unexpected form-form combination: {form0_type}, {form1_type}")

        # Get function and call
        func_name = func_template.format(suffix)
        cpp_func = getattr(_trueform.spatial, func_name)
        result = cpp_func(form0_obj._wrapper, form1_obj._wrapper, radius)

        # If forms were swapped, swap results back
        if result is not None and (needs_swap or extra_swap):
            (idx0, idx1), (dist, pt0, pt1) = result
            result = ((idx1, idx0), (dist, pt1, pt0))

        return result

    # Form-Primitive neighbor search (original implementation)
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
    obj_dims = spatial_object.dims
    if dims is None:
        raise TypeError(f"Cannot determine dimensions for query type {query_type}")
    if obj_dims != dims:
        raise ValueError(
            f"Dimension mismatch: spatial_object has {obj_dims}D, query has {dims}D. "
            f"Both must have the same dimensionality (2D or 3D)."
        )

    # Determine object type and compute appropriate suffix
    obj_type = type(spatial_object).__name__

    if obj_type == 'PointCloud':
        # PointCloud: suffix is "float2d" or "double3d"
        obj_dtype = spatial_object.points.dtype
        dtype_str = 'float' if obj_dtype == np.float32 else 'double'
        suffix = f"{dtype_str}{obj_dims}d"

    elif obj_type == 'Mesh':
        # Mesh: suffix is "intfloat32d" or "int64double43d"
        # Format: {index_type}{real_type}{ngon}{dims}d
        faces_dtype = spatial_object.faces.dtype
        points_dtype = spatial_object.points.dtype
        ngon = spatial_object.ngon

        index_str = 'int' if faces_dtype == np.int32 else 'int64'
        real_str = 'float' if points_dtype == np.float32 else 'double'
        suffix = f"{index_str}{real_str}{ngon}{obj_dims}d"
        obj_dtype = points_dtype  # Use points dtype for query conversion

    elif obj_type == 'EdgeMesh':
        # EdgeMesh: suffix is "intfloat2d" or "int64double3d"
        # Format: {index_type}{real_type}{dims}d
        edges_dtype = spatial_object.edges.dtype
        points_dtype = spatial_object.points.dtype

        index_str = 'int' if edges_dtype == np.int32 else 'int64'
        real_str = 'float' if points_dtype == np.float32 else 'double'
        suffix = f"{index_str}{real_str}{obj_dims}d"
        obj_dtype = points_dtype  # Use points dtype for query conversion

    else:
        raise TypeError(
            f"neighbor_search not implemented for spatial object type: {obj_type}. "
            f"Supported types: PointCloud, Mesh, EdgeMesh"
        )

    # Convert query_data to match object dtype if necessary
    if isinstance(query_data, np.ndarray) and query_data.dtype != obj_dtype:
        query_data = query_data.astype(obj_dtype)

    # Get the appropriate dispatch table for this object type
    if obj_type not in _DISPATCH_BY_TYPE:
        raise TypeError(
            f"neighbor_search not implemented for spatial object type: {obj_type}. "
            f"Supported types: {', '.join(_DISPATCH_BY_TYPE.keys())}"
        )

    # Choose dispatch table based on whether k is specified
    if k is None:
        # Non-KNN query - single nearest neighbor
        dispatch_table = _DISPATCH_BY_TYPE[obj_type]['single']

        if query_type not in dispatch_table:
            supported = ", ".join(t.__name__ for t in dispatch_table.keys())
            raise TypeError(
                f"neighbor_search not implemented for query type: {query_type.__name__}. "
                f"Supported types: {supported}"
            )

        func_name = dispatch_table[query_type].format(suffix)
        cpp_func = getattr(_trueform.spatial, func_name)
        return cpp_func(spatial_object._wrapper, query_data, radius)
    else:
        # KNN query - k nearest neighbors
        dispatch_table = _DISPATCH_BY_TYPE[obj_type]['knn']

        if query_type not in dispatch_table:
            supported = ", ".join(t.__name__ for t in dispatch_table.keys())
            raise TypeError(
                f"neighbor_search (KNN) not implemented for query type: {query_type.__name__}. "
                f"Supported types: {supported}"
            )

        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")

        func_name = dispatch_table[query_type].format(suffix)
        cpp_func = getattr(_trueform.spatial, func_name)
        return cpp_func(spatial_object._wrapper, query_data, k, radius)


__all__ = ['neighbor_search', 'gather_intersecting_ids', 'gather_ids_within_distance']
