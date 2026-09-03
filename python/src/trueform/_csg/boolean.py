"""
Boolean operations on meshes

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Optional, Sequence, Tuple, Union
from .._spatial import Mesh
from .._core import OffsetBlockedArray, as_offset_blocked
from .csg_graph import CsgGraph
from .expr import op


_OP_UNION = 0
_OP_INTERSECTION = 1
_OP_DIFFERENCE = 2

# The graph expression each pairwise operation answers.
_OP_EXPR = {
    _OP_UNION: lambda: op(0) | op(1),
    _OP_INTERSECTION: lambda: op(0) & op(1),
    _OP_DIFFERENCE: lambda: op(0) - op(1),
}


def boolean_union(
    mesh0: Mesh,
    mesh1: Mesh,
    return_curves: bool = False,
    sheets: Optional[Sequence[int]] = None
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
    sheets : sequence of int, optional
        Operand ids (0, 1) that bound no volume and act as oriented
        separators — a cutting plane, a fault. A sheet still cuts and is
        still cut, so difference and intersection against one give the two
        capped halves. This is a declaration of intent, not a property read
        off the mesh.
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
    face_labels : np.ndarray
        Labels indicating which source face each face came from, shape (N,)
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
    >>> (faces, points), labels, face_labels = tf.boolean_union(mesh0, mesh1)
    >>> print(f"Union has {len(faces)} faces")
    >>>
    >>> # Compute union with curves
    >>> (faces, points), labels, face_labels, (paths, curve_pts) = tf.boolean_union(
    ...     mesh0, mesh1, return_curves=True
    ... )
    """
    return _boolean_impl(mesh0, mesh1, _OP_UNION, return_curves, sheets)


def boolean_intersection(
    mesh0: Mesh,
    mesh1: Mesh,
    return_curves: bool = False,
    sheets: Optional[Sequence[int]] = None
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
    sheets : sequence of int, optional
        Operand ids (0, 1) that bound no volume and act as oriented
        separators — a cutting plane, a fault. A sheet still cuts and is
        still cut, so difference and intersection against one give the two
        capped halves. This is a declaration of intent, not a property read
        off the mesh.
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
    face_labels : np.ndarray
        Labels indicating which source face each face came from, shape (N,)
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
    >>> (faces, points), labels, face_labels = tf.boolean_intersection(mesh0, mesh1)
    >>> print(f"Intersection has {len(faces)} faces")
    """
    return _boolean_impl(mesh0, mesh1, _OP_INTERSECTION, return_curves, sheets)


def boolean_difference(
    mesh0: Mesh,
    mesh1: Mesh,
    return_curves: bool = False,
    sheets: Optional[Sequence[int]] = None
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
    face_labels : np.ndarray
        Labels indicating which source face each face came from, shape (N,)
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
    >>> (faces, points), labels, face_labels = tf.boolean_difference(mesh0, mesh1)
    >>> print(f"Difference has {len(faces)} faces")
    >>>
    >>> # Compute reverse difference mesh1 - mesh0
    >>> (faces, points), labels, face_labels = tf.boolean_difference(mesh1, mesh0)
    """
    return _boolean_impl(mesh0, mesh1, _OP_DIFFERENCE, return_curves, sheets)


def _boolean_impl(mesh0, mesh1, op_int, return_curves, sheets=None):
    """
    Internal implementation for boolean operations.

    Validates the pair, normalizes it to one representation, and answers
    the operation as a CsgGraph expression.
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

    # 5. VALIDATE SHEETS
    sheet_ids = [int(s) for s in (sheets or ())]
    for s in sheet_ids:
        if s not in (0, 1):
            raise ValueError(f"sheets must contain operand ids 0 or 1, got {s}")

    # 6. COMPOSE THROUGH THE GRAPH
    # The graph takes a homogeneous operand set, so a mixed pair is
    # normalized to one representation first: triangle faces re-expressed
    # as dynamic blocks, a narrower index dtype widened to int64. Both are
    # a faces-representation copy; points and transformations carry over.
    mesh0, mesh1 = _normalized_pair(mesh0, mesh1)
    graph = CsgGraph([mesh0, mesh1], sheets=sheet_ids)
    (faces, points), tag_labels, face_labels = graph.mesh(
        _OP_EXPR[op_int](), return_source_ids=True)
    labels = tag_labels.astype(np.int8)

    if return_curves:
        paths, curve_points = graph.intersection_curves()
        return (faces, points), labels, face_labels, (paths, curve_points)
    return (faces, points), labels, face_labels


def _carrying_transformation(mesh, source):
    if source.transformation is not None:
        mesh.transformation = source.transformation
    return mesh


def _as_dynamic(mesh):
    """The same mesh with triangle faces re-expressed as dynamic blocks."""
    return _carrying_transformation(
        Mesh(as_offset_blocked(mesh.faces), mesh.points), mesh)


def _widened(mesh):
    """The same mesh with its faces widened to int64."""
    if mesh.is_dynamic:
        faces = OffsetBlockedArray(mesh.faces.offsets.astype(np.int64),
                                   mesh.faces.data.astype(np.int64))
    else:
        faces = mesh.faces.astype(np.int64)
    return _carrying_transformation(Mesh(faces, mesh.points), mesh)


def _normalized_pair(mesh0, mesh1):
    if mesh0.is_dynamic != mesh1.is_dynamic:
        mesh0 = mesh0 if mesh0.is_dynamic else _as_dynamic(mesh0)
        mesh1 = mesh1 if mesh1.is_dynamic else _as_dynamic(mesh1)
    if mesh0.faces.dtype != mesh1.faces.dtype:
        wide = np.dtype(np.int64)
        mesh0 = mesh0 if mesh0.faces.dtype == wide else _widened(mesh0)
        mesh1 = mesh1 if mesh1.faces.dtype == wide else _widened(mesh1)
    return mesh0, mesh1
