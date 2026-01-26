"""
Unified gather_ids API

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Any, Optional
from .. import _trueform

# Dispatch infrastructure
from .._dispatch import (
    extract_meta,
    build_suffix,
    build_suffix_pair,
    canonicalize_index_order,
)
from ._dispatch import GATHER_IDS_FORM_PRIM, GATHER_IDS_FORM_FORM


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
        raise ValueError(
            f"Invalid predicate '{predicate}'. Must be 'intersects' or 'within_distance'")

    # Validate distance for within_distance
    if predicate == "within_distance" and distance is None:
        raise ValueError(
            "distance is required when predicate='within_distance'")

    # Validate dimensions match
    if not hasattr(form, 'dims') or not hasattr(query, 'dims'):
        raise TypeError("Both form and query must have 'dims' attribute")

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

    # Check if this is Form-Form or Form-Primitive operation
    both_forms = (form_type in {Mesh, EdgeMesh, PointCloud} and
                  query_type in {Mesh, EdgeMesh, PointCloud})

    if both_forms:
        return _form_form_gather_ids(form, query, type_pair, predicate, distance)
    else:
        return _form_prim_gather_ids(form, query, type_pair, predicate, distance)


def _form_prim_gather_ids(form, query, type_pair, predicate, distance):
    """Form x primitive gather_ids."""

    if type_pair not in GATHER_IDS_FORM_PRIM:
        supported = {t.__name__ for pair in GATHER_IDS_FORM_PRIM.keys()
                     for t in pair}
        raise TypeError(
            f"gather_ids not implemented for types: {type_pair[0].__name__}, {type_pair[1].__name__}. "
            f"Supported types: {', '.join(sorted(supported))}"
        )

    func_template, needs_swap = GATHER_IDS_FORM_PRIM[type_pair]

    # Determine form and primitive based on swap
    form_obj = query if needs_swap else form
    prim_obj = form if needs_swap else query

    # Build suffix using dispatch utility
    meta = extract_meta(form_obj)
    suffix = build_suffix(meta)
    func_name = func_template.format(suffix)
    cpp_func = getattr(_trueform.spatial, func_name)

    return cpp_func(form_obj._wrapper, prim_obj.data, predicate, distance)


def _form_form_gather_ids(form, query, type_pair, predicate, distance):
    """Form x form gather_ids."""

    if type_pair not in GATHER_IDS_FORM_FORM:
        supported = {t.__name__ for pair in GATHER_IDS_FORM_FORM.keys()
                     for t in pair}
        raise TypeError(
            f"gather_ids not implemented for types: {type_pair[0].__name__}, {type_pair[1].__name__}. "
            f"Supported form-form types: {', '.join(sorted(supported))}"
        )

    func_template, needs_swap = GATHER_IDS_FORM_FORM[type_pair]

    # Apply dispatch table swap
    form0_obj = query if needs_swap else form
    form1_obj = form if needs_swap else query

    # Apply index canonicalization (int32 before int64 for same types)
    form0_obj, form1_obj, extra_swap = canonicalize_index_order(
        form0_obj, form1_obj)

    # Build suffix using dispatch utility
    meta0 = extract_meta(form0_obj)
    meta1 = extract_meta(form1_obj)
    suffix = build_suffix_pair(meta0, meta1)

    func_name = func_template.format(suffix)
    cpp_func = getattr(_trueform.spatial, func_name)
    result = cpp_func(form0_obj._wrapper, form1_obj._wrapper,
                      predicate, distance)

    # If forms were swapped, swap result columns back
    # Result is numpy array of shape (N, 2) with columns [id0, id1]
    if result is not None and result.shape[0] > 0 and (needs_swap or extra_swap):
        result = result[:, [1, 0]]

    return result


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
