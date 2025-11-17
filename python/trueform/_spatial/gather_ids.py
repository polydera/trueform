"""
Unified gather_ids API

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

import numpy as np
from typing import Any, Optional
from .. import _trueform
from ._gather_ids import _GATHER_IDS_DISPATCH
from ._form_form_gather_ids import _FORM_FORM_GATHER_IDS_DISPATCH


def _gather_ids(form: Any, query: Any, predicate: str = "intersects", distance: Optional[float] = None) -> np.ndarray:
    """
    Gather IDs of primitives in a spatial form that satisfy a predicate with respect to a query primitive.

    This function searches a spatial data structure (Mesh, EdgeMesh, PointCloud) and returns the indices
    of all primitives that satisfy the given spatial predicate with respect to the query primitive.

    Parameters
    ----------
    form : Mesh, EdgeMesh, or PointCloud
        The spatial data structure to search
    query : Point, Segment, Polygon, Ray, or Line
        The query primitive
    predicate : str, optional
        The spatial predicate to evaluate. Options:
        - "intersects": Return primitives that intersect the query
        - "within_distance": Return primitives within distance of query
        Default is "intersects"
    distance : float, optional
        Distance for "within_distance" predicate. Required when predicate="within_distance"

    Returns
    -------
    numpy.ndarray
        Array of primitive indices that satisfy the predicate. Dtype matches the form's index type.

    Raises
    ------
    TypeError
        If form or query types are not supported
    ValueError
        If dimensionality doesn't match, or if distance is missing for "within_distance"

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Create a mesh
    >>> faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    >>> points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    >>> mesh = tf.Mesh(faces, points)
    >>>
    >>> # Find faces intersecting a point
    >>> pt = tf.Point([0.3, 0.3])
    >>> ids = tf.gather_ids(mesh, pt, predicate="intersects")
    >>> print(ids)  # e.g., [0]
    >>>
    >>> # Find faces within distance of a point
    >>> pt = tf.Point([2.0, 2.0])
    >>> ids = tf.gather_ids(mesh, pt, predicate="within_distance", distance=2.0)
    >>> print(ids)  # e.g., [1]
    """

    # Validate predicate
    if predicate not in ("intersects", "within_distance"):
        raise ValueError(f"Invalid predicate '{predicate}'. Must be 'intersects' or 'within_distance'")

    # Validate distance for within_distance
    if predicate == "within_distance" and distance is None:
        raise ValueError("distance is required when predicate='within_distance'")

    # Validate dimensions match
    if not hasattr(form, 'dims') or not hasattr(query, 'dims'):
        raise TypeError(f"Both form and query must have 'dims' attribute")

    if form.dims != query.dims:
        raise ValueError(
            f"Dimension mismatch: form has {form.dims}D, query has {query.dims}D. "
            f"Both objects must have the same dimensionality (2D or 3D)."
        )

    # Get types
    form_type = type(form)
    query_type = type(query)
    type_pair = (form_type, query_type)

    # Import spatial form types
    from .mesh import Mesh
    from .edge_mesh import EdgeMesh
    from .point_cloud import PointCloud

    # Check if this is a valid operation
    if type_pair not in _GATHER_IDS_DISPATCH:
        supported_forms = {Mesh, EdgeMesh, PointCloud}
        from .._primitives import Point, Segment, Polygon, Ray, Line
        supported_primitives = {Point, Segment, Polygon, Ray, Line}

        if form_type not in supported_forms:
            raise TypeError(
                f"gather_ids requires form to be Mesh, EdgeMesh, or PointCloud, got {form_type.__name__}"
            )
        if query_type not in supported_primitives and query_type not in supported_forms:
            raise TypeError(
                f"gather_ids requires query to be Point, Segment, Polygon, Ray, Line, Mesh, EdgeMesh, or PointCloud, got {query_type.__name__}"
            )

        raise TypeError(
            f"gather_ids not implemented for combination: {form_type.__name__}, {query_type.__name__}"
        )

    func_template, needs_swap = _GATHER_IDS_DISPATCH[type_pair]

    # Check if this is Form-Form or Form-Primitive operation
    both_forms = (form_type in {Mesh, EdgeMesh, PointCloud} and
                  query_type in {Mesh, EdgeMesh, PointCloud})

    if both_forms:
        # Form-Form operation - use dedicated dispatch table
        func_template, needs_swap = _FORM_FORM_GATHER_IDS_DISPATCH[type_pair]

        # Determine the canonical order
        form0_obj = form if not needs_swap else query
        form1_obj = query if not needs_swap else form
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
            suffix = f"{index0_str}{index1_str}{form0_obj.ngon}{real_str}{dims_str}"
        elif form0_type is Mesh and form1_type is Mesh:
            index0_str = 'int' if form0_obj.faces.dtype == np.int32 else 'int64'
            index1_str = 'int' if form1_obj.faces.dtype == np.int32 else 'int64'
            suffix = f"{index0_str}{index1_str}{form0_obj.ngon}{form1_obj.ngon}{real_str}{dims_str}"
        else:
            raise TypeError(f"Unexpected form-form combination: {form0_type}, {form1_type}")

        # Get function and call
        func_name = func_template.format(suffix)
        cpp_func = getattr(_trueform.spatial, func_name)
        result = cpp_func(form0_obj._wrapper, form1_obj._wrapper, predicate, distance)

        # If forms were swapped, swap result columns back
        # Result is numpy array of shape (N, 2) with columns [id0, id1]
        if result is not None and result.shape[0] > 0 and (needs_swap or extra_swap):
            result = result[:, [1, 0]]

        return result
    else:
        # Form-Primitive operation
        form_obj = form if not needs_swap else query
        prim_obj = query if not needs_swap else form

        # Compute suffix based on form type
        form_type = type(form_obj)
        if form_type is PointCloud:
            dtype_str = 'float' if form_obj.points.dtype == np.float32 else 'double'
            suffix = f"{dtype_str}{form_obj.dims}d"
        elif form_type is Mesh:
            faces_dtype = form_obj.faces.dtype
            points_dtype = form_obj.points.dtype
            index_str = 'int' if faces_dtype == np.int32 else 'int64'
            real_str = 'float' if points_dtype == np.float32 else 'double'
            suffix = f"{index_str}{real_str}{form_obj.ngon}{form_obj.dims}d"
        elif form_type is EdgeMesh:
            edges_dtype = form_obj.edges.dtype
            points_dtype = form_obj.points.dtype
            index_str = 'int' if edges_dtype == np.int32 else 'int64'
            real_str = 'float' if points_dtype == np.float32 else 'double'
            suffix = f"{index_str}{real_str}{form_obj.dims}d"
        else:
            raise TypeError(f"Unexpected form type: {form_type}")

        # Get function and call
        func_name = func_template.format(suffix)
        cpp_func = getattr(_trueform.spatial, func_name)

        # Call with predicate and distance
        return cpp_func(form_obj._wrapper, prim_obj.data, predicate, distance)


def gather_intersecting_ids(form: Any, query: Any) -> np.ndarray:
    """
    Gather IDs of primitives in a spatial form that intersect with a query.

    This function searches a spatial data structure (Mesh, EdgeMesh, PointCloud) and returns the indices
    of all primitives that intersect the query primitive or form.

    Parameters
    ----------
    form : Mesh, EdgeMesh, or PointCloud
        The spatial data structure to search
    query : Point, Segment, Polygon, Ray, Line, Mesh, EdgeMesh, or PointCloud
        The query primitive or form

    Returns
    -------
    numpy.ndarray
        Array of primitive indices that intersect the query.
        For form-form queries, returns shape (N, 2) with pairs of indices.
        For form-primitive queries, returns shape (N,) with single indices.
        Dtype matches the form's index type.

    Raises
    ------
    TypeError
        If form or query types are not supported
    ValueError
        If dimensionality doesn't match

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Create a mesh
    >>> faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    >>> points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    >>> mesh = tf.Mesh(faces, points)
    >>>
    >>> # Find faces intersecting a point
    >>> pt = tf.Point([0.3, 0.3])
    >>> ids = tf.gather_intersecting_ids(mesh, pt)
    >>> print(ids)  # e.g., [0]
    """
    return _gather_ids(form, query, predicate="intersects", distance=None)


def gather_ids_within_distance(form: Any, query: Any, distance: float) -> np.ndarray:
    """
    Gather IDs of primitives in a spatial form within a distance of a query.

    This function searches a spatial data structure (Mesh, EdgeMesh, PointCloud) and returns the indices
    of all primitives within the specified distance of the query primitive or form.

    Parameters
    ----------
    form : Mesh, EdgeMesh, or PointCloud
        The spatial data structure to search
    query : Point, Segment, Polygon, Ray, Line, Mesh, EdgeMesh, or PointCloud
        The query primitive or form
    distance : float
        Maximum distance

    Returns
    -------
    numpy.ndarray
        Array of primitive indices within distance of the query.
        For form-form queries, returns shape (N, 2) with pairs of indices.
        For form-primitive queries, returns shape (N,) with single indices.
        Dtype matches the form's index type.

    Raises
    ------
    TypeError
        If form or query types are not supported
    ValueError
        If dimensionality doesn't match

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Create a mesh
    >>> faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    >>> points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    >>> mesh = tf.Mesh(faces, points)
    >>>
    >>> # Find faces within distance of a point
    >>> pt = tf.Point([2.0, 2.0])
    >>> ids = tf.gather_ids_within_distance(mesh, pt, distance=2.0)
    >>> print(ids)  # e.g., [1]
    """
    return _gather_ids(form, query, predicate="within_distance", distance=distance)
