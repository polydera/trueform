"""
Boolean operations on meshes

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Tuple, Union
from .. import _trueform
from .._spatial import Mesh
from .._core import OffsetBlockedArray
from .._dispatch import extract_meta, build_suffix_pair, canonicalize_index_order


# Operation type constants (map to C++ tf::boolean_op enum)
_OP_UNION = 0  # boolean_op::merge
_OP_INTERSECTION = 1  # boolean_op::intersection
_OP_DIFFERENCE = 2  # boolean_op::left_difference


def boolean_union(
    mesh0: Mesh,
    mesh1: Mesh,
    return_curves: bool = False
) -> Union[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray],
           Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, Tuple[OffsetBlockedArray, np.ndarray]]]:
    """
    Compute boolean union of two 3D meshes (A ∪ B).

    Combines both meshes into a single mesh representing their union.
    Supports both triangle meshes and dynamic (variable polygon size) meshes.

    Parameters
    ----------
    mesh0 : Mesh
        First 3D mesh with topology (triangle or dynamic)
    mesh1 : Mesh
        Second 3D mesh with topology (triangle or dynamic)
        Must have same real dtype (float32 or float64) as mesh0
    return_curves : bool, default False
        If True, also return the intersection curves between the meshes

    Returns
    -------
    result_faces : np.ndarray or OffsetBlockedArray
        Face indices of the union mesh. Returns np.ndarray with shape (N, 3)
        if both inputs are triangle meshes, otherwise OffsetBlockedArray.
    result_points : np.ndarray
        Point coordinates of the union mesh, shape (M, 3)
    labels : np.ndarray
        Labels indicating which source mesh each face came from, shape (N,)
        Values: 0=mesh0, 1=mesh1
    paths : OffsetBlockedArray, optional
        Only returned if return_curves=True
        Intersection curves as indices into curve_points
    curve_points : np.ndarray, optional
        Only returned if return_curves=True
        Curve point coordinates with shape (P, 3)

    Examples
    --------
    >>> import trueform as tf
    >>> # Load two meshes
    >>> mesh0 = tf.Mesh(*tf.read_stl("mesh0.stl"))
    >>> mesh1 = tf.Mesh(*tf.read_stl("mesh1.stl"))
    >>>
    >>> # Compute union
    >>> (faces, points), labels = tf.boolean_union(mesh0, mesh1)
    >>> print(f"Union has {len(faces)} faces")
    >>>
    >>> # Compute union with curves
    >>> (faces, points), labels, (paths, curve_pts) = tf.boolean_union(
    ...     mesh0, mesh1, return_curves=True
    ... )
    """
    return _boolean_impl(mesh0, mesh1, _OP_UNION, return_curves)


def boolean_intersection(
    mesh0: Mesh,
    mesh1: Mesh,
    return_curves: bool = False
) -> Union[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray],
           Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, Tuple[OffsetBlockedArray, np.ndarray]]]:
    """
    Compute boolean intersection of two 3D meshes (A ∩ B).

    Returns the mesh representing the volume common to both inputs.
    Supports both triangle meshes and dynamic (variable polygon size) meshes.

    Parameters
    ----------
    mesh0 : Mesh
        First 3D mesh with topology (triangle or dynamic)
    mesh1 : Mesh
        Second 3D mesh with topology (triangle or dynamic)
        Must have same real dtype (float32 or float64) as mesh0
    return_curves : bool, default False
        If True, also return the intersection curves between the meshes

    Returns
    -------
    result_faces : np.ndarray or OffsetBlockedArray
        Face indices of the intersection mesh. Returns np.ndarray with shape (N, 3)
        if both inputs are triangle meshes, otherwise OffsetBlockedArray.
    result_points : np.ndarray
        Point coordinates of the intersection mesh, shape (M, 3)
    labels : np.ndarray
        Labels indicating which source mesh each face came from, shape (N,)
        Values: 0=mesh0, 1=mesh1
    paths : OffsetBlockedArray, optional
        Only returned if return_curves=True
        Intersection curves as indices into curve_points
    curve_points : np.ndarray, optional
        Only returned if return_curves=True
        Curve point coordinates with shape (P, 3)

    Examples
    --------
    >>> import trueform as tf
    >>> # Load two meshes
    >>> mesh0 = tf.Mesh(*tf.read_stl("mesh0.stl"))
    >>> mesh1 = tf.Mesh(*tf.read_stl("mesh1.stl"))
    >>>
    >>> # Compute intersection
    >>> (faces, points), labels = tf.boolean_intersection(mesh0, mesh1)
    >>> print(f"Intersection has {len(faces)} faces")
    """
    return _boolean_impl(mesh0, mesh1, _OP_INTERSECTION, return_curves)


def boolean_difference(
    mesh0: Mesh,
    mesh1: Mesh,
    return_curves: bool = False
) -> Union[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray],
           Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, Tuple[OffsetBlockedArray, np.ndarray]]]:
    """
    Compute boolean difference of two 3D meshes (A - B).

    Returns the mesh representing the volume in mesh0 that is not in mesh1.
    Supports both triangle meshes and dynamic (variable polygon size) meshes.

    Note: For the reverse operation (B - A), swap the arguments:
    `boolean_difference(mesh1, mesh0)`.

    Parameters
    ----------
    mesh0 : Mesh
        First 3D mesh with topology (triangle or dynamic, the mesh to subtract from)
    mesh1 : Mesh
        Second 3D mesh with topology (triangle or dynamic, the mesh to subtract)
        Must have same real dtype (float32 or float64) as mesh0
    return_curves : bool, default False
        If True, also return the intersection curves between the meshes

    Returns
    -------
    result_faces : np.ndarray or OffsetBlockedArray
        Face indices of the difference mesh. Returns np.ndarray with shape (N, 3)
        if both inputs are triangle meshes, otherwise OffsetBlockedArray.
    result_points : np.ndarray
        Point coordinates of the difference mesh, shape (M, 3)
    labels : np.ndarray
        Labels indicating which source mesh each face came from, shape (N,)
        Values: 0=mesh0, 1=mesh1
    paths : OffsetBlockedArray, optional
        Only returned if return_curves=True
        Intersection curves as indices into curve_points
    curve_points : np.ndarray, optional
        Only returned if return_curves=True
        Curve point coordinates with shape (P, 3)

    Examples
    --------
    >>> import trueform as tf
    >>> # Load two meshes
    >>> mesh0 = tf.Mesh(*tf.read_stl("mesh0.stl"))
    >>> mesh1 = tf.Mesh(*tf.read_stl("mesh1.stl"))
    >>>
    >>> # Compute difference mesh0 - mesh1
    >>> (faces, points), labels = tf.boolean_difference(mesh0, mesh1)
    >>> print(f"Difference has {len(faces)} faces")
    >>>
    >>> # Compute reverse difference mesh1 - mesh0
    >>> (faces, points), labels = tf.boolean_difference(mesh1, mesh0)
    """
    return _boolean_impl(mesh0, mesh1, _OP_DIFFERENCE, return_curves)


def _boolean_impl(mesh0, mesh1, op_int, return_curves):
    """
    Internal implementation for boolean operations.

    Handles validation, index type symmetry, and dispatching to C++.
    """

    # 1. VALIDATE INPUTS ARE MESH OBJECTS
    if not isinstance(mesh0, Mesh):
        raise TypeError(
            f"mesh0 must be a Mesh object, got {type(mesh0).__name__}. "
            f"Topology information is required for boolean operations."
        )

    if not isinstance(mesh1, Mesh):
        raise TypeError(
            f"mesh1 must be a Mesh object, got {type(mesh1).__name__}. "
            f"Topology information is required for boolean operations."
        )

    # 2. VALIDATE BOTH ARE 3D
    if mesh0.dims != 3:
        raise ValueError(
            f"Boolean operations only support 3D meshes, got mesh0 with {mesh0.dims}D"
        )
    if mesh1.dims != 3:
        raise ValueError(
            f"Boolean operations only support 3D meshes, got mesh1 with {mesh1.dims}D"
        )

    # 3. VALIDATE BOTH ARE TRIANGLES OR DYNAMIC
    if mesh0.ngon != 3 and not mesh0.is_dynamic:
        raise ValueError(
            f"Boolean operations only support triangle or dynamic meshes, got mesh0 with {mesh0.ngon}-gons"
        )
    if mesh1.ngon != 3 and not mesh1.is_dynamic:
        raise ValueError(
            f"Boolean operations only support triangle or dynamic meshes, got mesh1 with {mesh1.ngon}-gons"
        )

    # 4. VALIDATE REAL DTYPES MATCH
    if mesh0.dtype != mesh1.dtype:
        raise ValueError(
            f"Mesh dtypes must match: mesh0 has {mesh0.dtype}, mesh1 has {mesh1.dtype}. "
            f"Convert both meshes to the same dtype (float32 or float64)."
        )

    # 5. HANDLE INDEX TYPE SYMMETRY
    # C++ only implements: int×int, int×int64, int64×int64
    # If we have int64×int, swap to int×int64
    mesh0, mesh1, swapped = canonicalize_index_order(mesh0, mesh1)

    # 6. BUILD SUFFIX FOR C++ FUNCTION
    meta0 = extract_meta(mesh0)
    meta1 = extract_meta(mesh1)
    suffix = build_suffix_pair(meta0, meta1)

    # Determine if result will be dynamic (if either input is dynamic)
    result_is_dynamic = mesh0.is_dynamic or mesh1.is_dynamic

    # 7. DISPATCH TO C++
    if return_curves:
        func_name = f"boolean_curves_mesh_mesh_{suffix}"
        (result_faces, result_points), labels, ((paths_offsets, paths_data), curve_points) = getattr(
            _trueform.cut, func_name
        )(mesh0._wrapper, mesh1._wrapper, op_int)

        # 8. HANDLE LABEL SWAPPING
        # CRITICAL: If we swapped meshes, flip labels (0↔1)
        if swapped:
            labels = 1 - labels

        # Wrap result faces in OffsetBlockedArray if dynamic
        if result_is_dynamic:
            result_faces = OffsetBlockedArray(result_faces[0], result_faces[1])

        # Wrap paths in OffsetBlockedArray
        paths = OffsetBlockedArray(paths_offsets, paths_data)

        return (result_faces, result_points), labels, (paths, curve_points)
    else:
        func_name = f"boolean_mesh_mesh_{suffix}"
        (result_faces, result_points), labels = getattr(_trueform.cut, func_name)(
            mesh0._wrapper, mesh1._wrapper, op_int
        )

        # 8. HANDLE LABEL SWAPPING
        # CRITICAL: If we swapped meshes, flip labels (0↔1)
        if swapped:
            labels = 1 - labels

        # Wrap result faces in OffsetBlockedArray if dynamic
        if result_is_dynamic:
            result_faces = OffsetBlockedArray(result_faces[0], result_faces[1])

        return (result_faces, result_points), labels
