"""
Embed self-intersection curves into mesh topology

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Tuple, Union
from .. import _trueform
from .._spatial import Mesh
from .._core import OffsetBlockedArray
from .._dispatch import InputMeta, build_suffix

_MODE_MAP = {"sos": 1, "primitives": 2}
_RESOLVE_CROSSINGS = 4
_RESOLVE_SELF_CROSSINGS = 8


def embedded_self_intersection_curves(
    mesh: Mesh,
    return_curves: bool = False,
    *,
    mode: str = "primitives",
    resolve_crossings: bool = True,
    resolve_self_crossings: bool = True
) -> Union[Tuple[np.ndarray, np.ndarray],
           Tuple[Tuple[np.ndarray, np.ndarray], Tuple[OffsetBlockedArray, np.ndarray]]]:
    """
    Embed self-intersection curves into mesh topology.

    Splits faces along self-intersection curves so that the intersections
    become edges in the resulting mesh. This is useful for mesh repair
    workflows where self-intersections need to be resolved.
    Supports both triangle meshes and dynamic (variable polygon size) meshes.

    Parameters
    ----------
    mesh : Mesh
        3D mesh with potential self-intersections (triangle or dynamic).
    return_curves : bool, default False
        If True, also return the self-intersection curves.
    mode : str, default "primitives"
        Intersection mode. "sos" or "primitives".
    resolve_crossings : bool, default True
        Resolve crossings between different contours on the same face.
    resolve_self_crossings : bool, default True
        Resolve self-crossings within a single contour.

    Returns
    -------
    result_faces : np.ndarray or OffsetBlockedArray
        Face indices of the result mesh. Returns np.ndarray with shape (N, 3)
        if input is triangle mesh, otherwise OffsetBlockedArray.
    result_points : np.ndarray
        Point coordinates of the result mesh, shape (M, 3)
    paths : OffsetBlockedArray, optional
        Only returned if return_curves=True.
        Self-intersection curves as indices into curve_points.
    curve_points : np.ndarray, optional
        Only returned if return_curves=True.
        Curve point coordinates with shape (P, 3).

    Raises
    ------
    ValueError
        If mesh is not 3D.
    TypeError
        If input is not a Mesh object.

    Examples
    --------
    >>> import trueform as tf
    >>> mesh = tf.Mesh(faces, points)
    >>> result_faces, result_points = tf.embedded_self_intersection_curves(mesh)
    >>> (faces, points), (paths, curve_pts) = tf.embedded_self_intersection_curves(
    ...     mesh, return_curves=True
    ... )

    Notes
    -----
    The input mesh remains unchanged. The output mesh has faces split such
    that no face contains a self-intersection curve in its interior — all
    self-intersections become edges.
    """

    if not isinstance(mesh, Mesh):
        raise TypeError(
            f"mesh must be a Mesh object, got {type(mesh).__name__}. "
            f"Topology information is required for embedded_self_intersection_curves."
        )

    if mesh.dims != 3:
        raise ValueError(
            f"embedded_self_intersection_curves only supports 3D meshes, got mesh with {mesh.dims}D"
        )

    if mesh.ngon != 3 and not mesh.is_dynamic:
        raise ValueError(
            f"embedded_self_intersection_curves only supports triangle or dynamic meshes, got ngon={mesh.ngon}"
        )

    if mode not in _MODE_MAP:
        raise ValueError(f"mode must be 'sos' or 'primitives', got '{mode}'")

    mode_int = _MODE_MAP[mode]
    if resolve_crossings:
        mode_int |= _RESOLVE_CROSSINGS
    if resolve_self_crossings:
        mode_int |= _RESOLVE_SELF_CROSSINGS

    ngon = 'dyn' if mesh.is_dynamic else '3'
    meta = InputMeta(mesh.faces.dtype, mesh.dtype, ngon, 3)
    suffix = build_suffix(meta)

    if return_curves:
        func_name = f"embedded_self_intersection_curves_curves_mesh_{suffix}"
        (result_faces, result_points), face_labels, ((paths_offsets, paths_data), curve_points) = getattr(
            _trueform.cut, func_name
        )(mesh._wrapper, mode_int)

        if mesh.is_dynamic:
            result_faces = OffsetBlockedArray(result_faces[0], result_faces[1])

        paths = OffsetBlockedArray(paths_offsets, paths_data)
        return (result_faces, result_points), face_labels, (paths, curve_points)
    else:
        func_name = f"embedded_self_intersection_curves_mesh_{suffix}"
        (result_faces, result_points), face_labels = getattr(_trueform.cut, func_name)(
            mesh._wrapper, mode_int
        )

        if mesh.is_dynamic:
            result_faces = OffsetBlockedArray(result_faces[0], result_faces[1])

        return (result_faces, result_points), face_labels
