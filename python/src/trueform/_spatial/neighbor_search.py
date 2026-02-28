"""
Neighbor search functions for spatial queries

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Any, Optional, Union, List, Tuple
from .. import _trueform
from .._primitives.primitive import Primitive
from .._primitives import Point
from .._dispatch import extract_meta, build_suffix, build_suffix_pair, canonicalize_index_order
from ._dispatch import NEIGHBOR_SEARCH_FORM_FORM


def neighbor_search(
    spatial_object: Any,
    query: Any,
    radius: Optional[float] = None,
    k: Optional[int] = None
) -> Union[Tuple[int, float, np.ndarray], List[Tuple[int, float, np.ndarray]], Tuple[Tuple[int, int], Tuple[float, np.ndarray, np.ndarray]]]:
    """
    Search for nearest neighbor(s) in a spatial structure.

    Performs spatial queries to find the closest element(s) in a point cloud, mesh, or edge mesh to a given
    geometric primitive (point, segment, triangle, polygon, ray, or line) or another spatial form.

    Parameters
    ----------
    spatial_object : PointCloud, Mesh, or EdgeMesh
        The spatial structure to search in
    query : Point, Segment, Triangle, Polygon, Ray, Line, PointCloud, Mesh, EdgeMesh, or numpy array
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
    single_result : tuple[int, float, ndarray] (single primitive, k=None)
        Returns (index, distance_squared, point) for the nearest neighbor,
        where index is the element index, distance_squared is the squared distance,
        and point is the coordinates of the closest point on the query primitive.
        Returns None if no neighbor found within radius.
    batch_result : tuple[ndarray, ndarray, ndarray] (batch primitive, k=None)
        Returns (ids[N], distances[N], points[N, D]) where ids contains element indices
        (-1 for not found), distances contains squared distances (inf for not found),
        and points contains closest points on the spatial structure.
    single_knn : list[tuple[int, float, ndarray]] (single primitive, k specified)
        Returns a list of up to k tuples (index, distance_squared, point)
        sorted by distance (closest first).
    batch_knn : tuple[ndarray, ndarray, ndarray, ndarray] (batch primitive, k specified)
        Returns (ids[N,K], distances[N,K], points[N,K,D], counts[N]) where
        ids contains element indices (-1 padding for unfilled slots),
        distances contains squared distances, points contains closest points,
        and counts[i] gives the number of valid neighbors for query i.
    form_form_result : tuple[tuple[int, int], tuple[float, ndarray, ndarray]] (form-form query)
        Returns ((index0, index1), (distance, point0, point1)) where
        index0 and index1 are element indices in the two forms, distance is the squared distance,
        and point0, point1 are the closest points on each form.

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
    >>> # Batch query: find nearest neighbor for 100 query points
    >>> query_pts = tf.Point(np.random.rand(100, 3).astype(np.float32))
    >>> ids, dists, pts = tf.neighbor_search(cloud, query_pts)
    >>> print(f"ids shape: {ids.shape}, dists shape: {dists.shape}")
    >>>
    >>> # Batch k-NN: find 3 nearest neighbors for 100 query points
    >>> ids, dists, pts, counts = tf.neighbor_search(cloud, query_pts, k=3)
    >>> print(f"ids shape: {ids.shape}, counts: {counts}")  # (100, 3), (100,)
    >>>
    >>> # Form-form neighbor search
    >>> cloud2 = tf.PointCloud(points + 0.5)
    >>> (idx0, idx1), (dist, pt0, pt1) = tf.neighbor_search(cloud, cloud2)
    >>> print(f"Closest pair: cloud[{idx0}] to cloud2[{idx1}], distance²={dist}")
    """
    from .mesh import Mesh
    from .edge_mesh import EdgeMesh
    from .point_cloud import PointCloud

    is_form_query = isinstance(query, (Mesh, EdgeMesh, PointCloud))

    if is_form_query:
        return _form_form_neighbor_search(spatial_object, query, radius, k)

    # Wrap raw numpy arrays as Point
    if isinstance(query, np.ndarray):
        if query.ndim == 1:
            query = Point(query)
        else:
            raise TypeError(
                f"numpy array queries must be 1D point arrays, got shape {query.shape}")

    if not isinstance(query, Primitive):
        raise TypeError(
            f"query must be a Primitive, got {type(query).__name__}")

    if spatial_object.dims != query.dims:
        raise ValueError(
            f"Dimension mismatch: spatial_object has {spatial_object.dims}D, "
            f"query has {query.dims}D. "
            f"Both must have the same dimensionality (2D or 3D).")

    if spatial_object.dtype != query.dtype:
        raise TypeError(
            f"Dtype mismatch: spatial_object has {spatial_object.dtype}, "
            f"query has {query.dtype}. "
            f"Both objects must have the same dtype.")

    meta = extract_meta(spatial_object)
    suffix = build_suffix(meta)

    if k is None:
        fn = getattr(_trueform.spatial,
                      f"neighbor_search_{meta.form_name}_fp_{suffix}")
        return fn(spatial_object._wrapper, query._wrapper, radius)
    else:
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        fn = getattr(_trueform.spatial,
                      f"neighbor_search_knn_{meta.form_name}_fp_{suffix}")
        return fn(spatial_object._wrapper, query._wrapper, k, radius)


def _form_form_neighbor_search(form0, form1, radius, k):
    """Form x form neighbor search using centralized dispatch."""

    if k is not None:
        raise ValueError("KNN (k parameter) is not supported for form-form neighbor_search")

    if form0.dims != form1.dims:
        raise ValueError(
            f"Dimension mismatch: first form has {form0.dims}D, "
            f"second form has {form1.dims}D. Both must have the same dimensionality (2D or 3D)."
        )

    form0_type = type(form0)
    form1_type = type(form1)
    type_pair = (form0_type, form1_type)

    if type_pair not in NEIGHBOR_SEARCH_FORM_FORM:
        supported = {t.__name__ for pair in NEIGHBOR_SEARCH_FORM_FORM.keys() for t in pair}
        raise TypeError(
            f"form-form neighbor_search not implemented for types: {form0_type.__name__}, {form1_type.__name__}. "
            f"Supported types: {', '.join(sorted(supported))}"
        )

    func_template, needs_swap = NEIGHBOR_SEARCH_FORM_FORM[type_pair]

    form0_obj = form1 if needs_swap else form0
    form1_obj = form0 if needs_swap else form1

    form0_obj, form1_obj, extra_swap = canonicalize_index_order(form0_obj, form1_obj)

    meta0 = extract_meta(form0_obj)
    meta1 = extract_meta(form1_obj)
    suffix = build_suffix_pair(meta0, meta1)

    func_name = func_template.format(suffix)
    cpp_func = getattr(_trueform.spatial, func_name)
    result = cpp_func(form0_obj._wrapper, form1_obj._wrapper, radius)

    if result is not None and (needs_swap or extra_swap):
        (idx0, idx1), (dist, pt0, pt1) = result
        result = ((idx1, idx0), (dist, pt1, pt0))

    return result
