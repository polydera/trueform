"""
Mesh arrangement operations

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
from .._dispatch import extract_meta, build_suffix


_MODE_MAP = {"sos": 1, "primitives": 2}
_RESOLVE_CROSSINGS = 4
_RESOLVE_SELF_CROSSINGS = 8
_SELF_INTERSECTIONS = 16

_TRIANGULATION_MAP = {"cdt": 0, "refined_cdt": 1}


def mesh_arrangements(
    meshes,
    *,
    return_curves: bool = False,
    mode: str = "primitives",
    tolerance: float = 0.0,
    resolve_crossings: bool = None,
    resolve_self_crossings: bool = False,
    within: bool = False,
    triangulation: str = "cdt"
):
    """
    Compute mesh arrangement from N meshes.

    Splits all faces along intersection curves and merges into one mesh.
    Each face is tagged with its source mesh index and original face index.

    Parameters
    ----------
    meshes : list of Mesh
        Two or more 3D triangle meshes. All must have the same real dtype
        (float32 or float64).
    return_curves : bool, default False
        If True, also return intersection curves.
    mode : str, default "primitives"
        Intersection mode. "sos" or "primitives".
    tolerance : float, default 0.0
        World-coordinate distance an input vertex may move to reach the lattice
        (0 = exact).
    resolve_crossings : bool, optional
        Resolve crossings between different contours on the same face.
        Default: False for 2 meshes, True for 3+ meshes.
    resolve_self_crossings : bool, default False
        Resolve self-crossings within a single contour.
    within : bool, default False
        Also intersect each mesh with itself. Required when a mesh can
        self-overlap, e.g. meshes concatenated into one input.
    triangulation : str, default "cdt"
        Cut-surface triangulation: "cdt" (plain constrained Delaunay per
        cut loop) or "refined_cdt" (quality-refined triangulation of the
        cut surface, adding Steiner points; shared boundaries stay
        watertight by construction).

    Returns
    -------
    result_faces : np.ndarray
        Face indices of the arrangement mesh, shape (N, 3)
    result_points : np.ndarray
        Point coordinates of the arrangement mesh, shape (M, 3)
    tag_labels : np.ndarray
        Per-face label: which input mesh each face came from, shape (N,)
    face_labels : np.ndarray
        Per-face label: which face in the original mesh, shape (N,)
    paths : OffsetBlockedArray, optional
        Only returned if return_curves=True.
        Intersection curve paths as indices into curve_points.
    curve_points : np.ndarray, optional
        Only returned if return_curves=True.
        Curve point coordinates, shape (P, 3).

    Examples
    --------
    >>> import trueform as tf
    >>> m0 = tf.Mesh(*tf.read_stl("mesh0.stl"))
    >>> m1 = tf.Mesh(*tf.read_stl("mesh1.stl"))
    >>> (faces, points), tag_labels, face_labels = tf.mesh_arrangements([m0, m1])
    """

    # Validate
    if not isinstance(meshes, (list, tuple)) or len(meshes) < 2:
        raise ValueError("mesh_arrangements requires a list of at least 2 meshes")

    for i, m in enumerate(meshes):
        if not isinstance(m, Mesh):
            raise TypeError(
                f"meshes[{i}] must be a Mesh object, got {type(m).__name__}"
            )
        if m.dims != 3:
            raise ValueError(
                f"meshes[{i}] must be 3D, got {m.dims}D"
            )
        if m.ngon != 3 and not m.is_dynamic:
            raise ValueError(
                f"meshes[{i}] must be triangle or dynamic, got {m.ngon}-gons"
            )

    # Validate all have same types
    meta0 = extract_meta(meshes[0])
    for i, m in enumerate(meshes[1:], 1):
        mi = extract_meta(m)
        if mi.real_dtype != meta0.real_dtype:
            raise ValueError(
                f"All meshes must have same real dtype: meshes[0] has {meta0.real_dtype}, "
                f"meshes[{i}] has {mi.real_dtype}"
            )
        if mi.index_dtype != meta0.index_dtype:
            raise ValueError(
                f"All meshes must have same index dtype: meshes[0] has {meta0.index_dtype}, "
                f"meshes[{i}] has {mi.index_dtype}"
            )
        if mi.ngon != meta0.ngon:
            raise ValueError(
                f"All meshes must have same ngon: meshes[0] has {meta0.ngon}, "
                f"meshes[{i}] has {mi.ngon}"
            )

    # Build suffix from first mesh
    meta = meta0
    suffix = build_suffix(meta)

    # All meshes must be same type (enforced above), so check first
    result_is_dynamic = meshes[0].is_dynamic

    # Build mode int
    if mode not in _MODE_MAP:
        raise ValueError(f"mode must be 'sos' or 'primitives', got '{mode}'")
    if triangulation not in _TRIANGULATION_MAP:
        raise ValueError(
            "triangulation must be 'cdt' or 'refined_cdt', "
            f"got '{triangulation}'"
        )
    triangulation_int = _TRIANGULATION_MAP[triangulation]
    mode_int = _MODE_MAP[mode]
    rc = resolve_crossings if resolve_crossings is not None else len(meshes) > 2
    if rc:
        mode_int |= _RESOLVE_CROSSINGS
    if resolve_self_crossings:
        mode_int |= _RESOLVE_SELF_CROSSINGS
    if within:
        mode_int |= _SELF_INTERSECTIONS | _RESOLVE_SELF_CROSSINGS

    # Extract wrappers
    wrappers = [m._wrapper for m in meshes]

    # Dispatch
    if return_curves:
        func_name = f"mesh_arrangements_curves_{suffix}"
        (result_faces, result_points), tag_labels, face_labels, \
            ((paths_offsets, paths_data), curve_points) = getattr(
                _trueform.arrangement, func_name
            )(wrappers, mode_int, tolerance, triangulation_int)
        if result_is_dynamic:
            result_faces = OffsetBlockedArray(result_faces[0], result_faces[1])
        paths = OffsetBlockedArray(paths_offsets, paths_data)
        return (result_faces, result_points), tag_labels, face_labels, \
            (paths, curve_points)
    else:
        func_name = f"mesh_arrangements_{suffix}"
        (result_faces, result_points), tag_labels, face_labels = getattr(
            _trueform.arrangement, func_name
        )(wrappers, mode_int, tolerance, triangulation_int)
        if result_is_dynamic:
            result_faces = OffsetBlockedArray(result_faces[0], result_faces[1])
        return (result_faces, result_points), tag_labels, face_labels
