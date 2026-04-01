"""
Polygon arrangement operations (self-intersection arrangements)

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Tuple
from .. import _trueform
from .._spatial import Mesh
from .._core import OffsetBlockedArray
from .._dispatch import InputMeta, build_suffix

_MODE_MAP = {"sos": 1, "primitives": 2}
_RESOLVE_CROSSINGS = 4
_RESOLVE_SELF_CROSSINGS = 8


def polygon_arrangements(
    mesh: Mesh,
    *,
    return_curves: bool = False,
    mode: str = "primitives",
    resolve_crossings: bool = True,
    resolve_self_crossings: bool = True
):
    """
    Compute polygon arrangement from a single mesh's self-intersections.

    Splits all faces along self-intersection curves and returns the
    subdivided mesh with per-face labels identifying connected regions.

    Parameters
    ----------
    mesh : Mesh
        3D mesh to decompose at self-intersections.
    return_curves : bool, default False
        If True, also return self-intersection curves.
    mode : str, default "primitives"
        Intersection mode. "sos" or "primitives".
    resolve_crossings : bool, default True
        Resolve crossings between different contours on the same face.
    resolve_self_crossings : bool, default True
        Resolve self-crossings within a single contour.

    Returns
    -------
    result_faces : np.ndarray or OffsetBlockedArray
        Face indices of the arrangement mesh, shape (N, 3).
    result_points : np.ndarray
        Point coordinates of the arrangement mesh, shape (M, 3).
    face_labels : np.ndarray
        Per-face label: which original face each output face came from, shape (N,).
    paths : OffsetBlockedArray, optional
        Only returned if return_curves=True.
        Self-intersection curve paths as indices into curve_points.
    curve_points : np.ndarray, optional
        Only returned if return_curves=True.
        Curve point coordinates, shape (P, 3).

    Examples
    --------
    >>> import trueform as tf
    >>> mesh = tf.Mesh(*tf.read_stl("self_intersecting.stl"))
    >>> (faces, points), face_labels = tf.polygon_arrangements(mesh)
    """

    if not isinstance(mesh, Mesh):
        raise TypeError(
            f"mesh must be a Mesh object, got {type(mesh).__name__}. "
            f"Topology information is required for polygon arrangements."
        )

    if mesh.dims != 3:
        raise ValueError(
            f"polygon_arrangements only supports 3D meshes, got mesh with {mesh.dims}D"
        )

    if mode not in _MODE_MAP:
        raise ValueError(f"mode must be 'sos' or 'primitives', got '{mode}'")

    mode_int = _MODE_MAP[mode]
    if resolve_crossings:
        mode_int |= _RESOLVE_CROSSINGS
    if resolve_self_crossings:
        mode_int |= _RESOLVE_SELF_CROSSINGS

    ngon = 'dyn' if mesh.is_dynamic else str(mesh.ngon)
    meta = InputMeta(mesh.faces.dtype, mesh.dtype, ngon, 3)
    suffix = build_suffix(meta)

    result_is_dynamic = mesh.is_dynamic

    if return_curves:
        func_name = f"polygon_arrangements_curves_{suffix}"
        (result_faces, result_points), face_labels, \
            ((paths_offsets, paths_data), curve_points) = getattr(
                _trueform.cut, func_name
            )(mesh._wrapper, mode_int)
        if result_is_dynamic:
            result_faces = OffsetBlockedArray(result_faces[0], result_faces[1])
        paths = OffsetBlockedArray(paths_offsets, paths_data)
        return (result_faces, result_points), face_labels, \
            (paths, curve_points)
    else:
        func_name = f"polygon_arrangements_{suffix}"
        (result_faces, result_points), face_labels = getattr(
            _trueform.cut, func_name
        )(mesh._wrapper, mode_int)
        if result_is_dynamic:
            result_faces = OffsetBlockedArray(result_faces[0], result_faces[1])
        return (result_faces, result_points), face_labels
