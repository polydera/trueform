"""
Embed intersection curves between two meshes into first mesh

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
from .._dispatch import extract_meta, build_suffix_pair

_MODE_MAP = {"sos": 1, "primitives": 2}
_RESOLVE_CROSSINGS = 4
_RESOLVE_SELF_CROSSINGS = 8


def embedded_intersection_curves(
    mesh0: Mesh,
    mesh1: Mesh,
    return_curves: bool = False,
    *,
    mode: str = "primitives",
    tolerance: float = 0.0,
    resolve_crossings: bool = False,
    resolve_self_crossings: bool = False
):
    """
    Embed intersection curves between mesh A and mesh B into mesh A.

    Splits faces of mesh0 along intersection curves with mesh1, so that the
    intersections become edges in the resulting mesh. All faces from mesh0
    are preserved (split where intersecting), with no faces from mesh1.

    Parameters
    ----------
    mesh0 : Mesh
        3D mesh to embed curves into (triangle or dynamic).
    mesh1 : Mesh
        3D mesh providing the cutting surface (triangle or dynamic).
        Must have same real dtype as mesh0.
    return_curves : bool, default False
        If True, also return the intersection curves.
    mode : str, default "primitives"
        Intersection mode. "sos" or "primitives".
    tolerance : float, default 0.0
        World-coordinate distance band for predicate tolerance (0 = exact).
    resolve_crossings : bool, default False
        Resolve crossings between different contours on the same face.
    resolve_self_crossings : bool, default False
        Resolve self-crossings within a single contour.

    Returns
    -------
    result_faces : np.ndarray or OffsetBlockedArray
        Face indices of the result mesh. Returns np.ndarray with shape (N, 3)
        if both inputs are triangle meshes, otherwise OffsetBlockedArray.
    result_points : np.ndarray
        Point coordinates of the result mesh, shape (M, 3).
    face_labels : np.ndarray
        Per-face origin: which face in the original mesh each output face
        came from. Shape (N,).
    paths : OffsetBlockedArray, optional
        Only returned if return_curves=True.
        Intersection curves as indices into curve_points.
    curve_points : np.ndarray, optional
        Only returned if return_curves=True.
        Curve point coordinates with shape (P, 3).
    """

    if not isinstance(mesh0, Mesh):
        raise TypeError(
            f"mesh0 must be a Mesh object, got {type(mesh0).__name__}. "
            f"Topology information is required for embedded_intersection_curves."
        )
    if not isinstance(mesh1, Mesh):
        raise TypeError(
            f"mesh1 must be a Mesh object, got {type(mesh1).__name__}. "
            f"Topology information is required for embedded_intersection_curves."
        )
    if mesh0.dims != 3:
        raise ValueError(
            f"embedded_intersection_curves only supports 3D meshes, got mesh0 with {mesh0.dims}D"
        )
    if mesh1.dims != 3:
        raise ValueError(
            f"embedded_intersection_curves only supports 3D meshes, got mesh1 with {mesh1.dims}D"
        )
    if mesh0.ngon != 3 and not mesh0.is_dynamic:
        raise ValueError(
            f"embedded_intersection_curves only supports triangle or dynamic meshes, got mesh0 with {mesh0.ngon}-gons"
        )
    if mesh1.ngon != 3 and not mesh1.is_dynamic:
        raise ValueError(
            f"embedded_intersection_curves only supports triangle or dynamic meshes, got mesh1 with {mesh1.ngon}-gons"
        )
    if mesh0.dtype != mesh1.dtype:
        raise ValueError(
            f"Mesh dtypes must match: mesh0 has {mesh0.dtype}, mesh1 has {mesh1.dtype}. "
            f"Convert both meshes to the same dtype (float32 or float64)."
        )

    if mode not in _MODE_MAP:
        raise ValueError(f"mode must be 'sos' or 'primitives', got '{mode}'")

    mode_int = _MODE_MAP[mode]
    if resolve_crossings:
        mode_int |= _RESOLVE_CROSSINGS
    if resolve_self_crossings:
        mode_int |= _RESOLVE_SELF_CROSSINGS

    meta0 = extract_meta(mesh0)
    meta1 = extract_meta(mesh1)
    suffix = build_suffix_pair(meta0, meta1)

    result_is_dynamic = mesh0.is_dynamic

    if return_curves:
        func_name = f"embedded_intersection_curves_curves_mesh_mesh_{suffix}"
        (result_faces, result_points), face_labels, ((paths_offsets, paths_data), curve_points) = getattr(
            _trueform.cut, func_name
        )(mesh0._wrapper, mesh1._wrapper, mode_int, tolerance)

        if result_is_dynamic:
            result_faces = OffsetBlockedArray(result_faces[0], result_faces[1])

        paths = OffsetBlockedArray(paths_offsets, paths_data)
        return (result_faces, result_points), face_labels, (paths, curve_points)
    else:
        func_name = f"embedded_intersection_curves_mesh_mesh_{suffix}"
        (result_faces, result_points), face_labels = getattr(_trueform.cut, func_name)(
            mesh0._wrapper, mesh1._wrapper, mode_int, tolerance
        )

        if result_is_dynamic:
            result_faces = OffsetBlockedArray(result_faces[0], result_faces[1])

        return (result_faces, result_points), face_labels
