"""
Intersection curves between meshes

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Tuple, Union, List
from .. import _trueform
from .._spatial import Mesh
from .._core import OffsetBlockedArray
from .._dispatch import extract_meta, build_suffix, build_suffix_pair, canonicalize_index_order

_MODE_MAP = {"sos": 1, "primitives": 2}
_RESOLVE_CROSSINGS = 4
_RESOLVE_SELF_CROSSINGS = 8


def _build_mode(mode: str, resolve_crossings: bool, resolve_self_crossings: bool) -> int:
    if mode not in _MODE_MAP:
        raise ValueError(f"mode must be 'sos' or 'primitives', got '{mode}'")
    m = _MODE_MAP[mode]
    if resolve_crossings:
        m |= _RESOLVE_CROSSINGS
    if resolve_self_crossings:
        m |= _RESOLVE_SELF_CROSSINGS
    return m


def intersection_curves(
    meshes_or_mesh0: Union[Mesh, List[Mesh]],
    mesh1: Mesh = None,
    *,
    mode: str = "sos",
    resolve_crossings: bool = None,
    resolve_self_crossings: bool = False
) -> Tuple[OffsetBlockedArray, np.ndarray]:
    """
    Compute intersection curves between meshes.

    Accepts either two meshes or a list of meshes. For two meshes, finds
    the curves where they intersect. For a list, finds all pairwise
    intersection curves.

    Parameters
    ----------
    meshes_or_mesh0 : Mesh or list of Mesh
        Either the first mesh (when mesh1 is provided), or a list of
        2+ meshes for N-mesh intersection.
    mesh1 : Mesh, optional
        Second mesh. Required when meshes_or_mesh0 is a single Mesh.
    mode : str, default "sos"
        Intersection mode. "sos" = SoS (fast), "primitives" = handles
        shared edges/vertices.
    resolve_crossings : bool, optional
        Resolve crossings between different contours on the same face.
        Default: False for 2-mesh, True for N-mesh.
    resolve_self_crossings : bool, default False
        Resolve self-crossings within a single contour.

    Returns
    -------
    paths : OffsetBlockedArray
        Paths as indices into the points array. Each path is one curve.
    points : np.ndarray
        Curve point coordinates with shape (N, 3).
    """

    if isinstance(meshes_or_mesh0, (list, tuple)):
        rc = resolve_crossings if resolve_crossings is not None else True
        m = _build_mode(mode, rc, resolve_self_crossings)
        return _intersection_curves_list(meshes_or_mesh0, mode=m)
    else:
        if mesh1 is None:
            raise ValueError(
                "intersection_curves requires either two meshes or a list of meshes"
            )
        rc = resolve_crossings if resolve_crossings is not None else False
        m = _build_mode(mode, rc, resolve_self_crossings)
        return _intersection_curves_pair(meshes_or_mesh0, mesh1, mode=m)


def _intersection_curves_pair(mesh0: Mesh, mesh1: Mesh, *, mode: int) -> Tuple[OffsetBlockedArray, np.ndarray]:
    if not isinstance(mesh0, Mesh):
        raise TypeError(
            f"mesh0 must be a Mesh object, got {type(mesh0).__name__}. "
            f"Topology information is required for intersection curves."
        )
    if not isinstance(mesh1, Mesh):
        raise TypeError(
            f"mesh1 must be a Mesh object, got {type(mesh1).__name__}. "
            f"Topology information is required for intersection curves."
        )
    if mesh0.dims != 3:
        raise ValueError(
            f"intersection_curves only supports 3D meshes, got mesh0 with {mesh0.dims}D"
        )
    if mesh1.dims != 3:
        raise ValueError(
            f"intersection_curves only supports 3D meshes, got mesh1 with {mesh1.dims}D"
        )
    if mesh0.dtype != mesh1.dtype:
        raise ValueError(
            f"Mesh dtypes must match: mesh0 has {mesh0.dtype}, mesh1 has {mesh1.dtype}. "
            f"Convert both meshes to the same dtype (float32 or float64)."
        )

    mesh0, mesh1, _ = canonicalize_index_order(mesh0, mesh1)
    meta0 = extract_meta(mesh0)
    meta1 = extract_meta(mesh1)
    suffix = build_suffix_pair(meta0, meta1)

    func_name = f"intersection_curves_mesh_mesh_{suffix}"
    (paths_offsets, paths_data), points = getattr(_trueform.intersect, func_name)(
        mesh0._wrapper, mesh1._wrapper, mode
    )
    return OffsetBlockedArray(paths_offsets, paths_data), points


def _intersection_curves_list(meshes: list, *, mode: int) -> Tuple[OffsetBlockedArray, np.ndarray]:
    if len(meshes) < 2:
        raise ValueError("intersection_curves requires at least 2 meshes")

    for i, m in enumerate(meshes):
        if not isinstance(m, Mesh):
            raise TypeError(f"meshes[{i}] must be a Mesh object, got {type(m).__name__}")
        if m.dims != 3:
            raise ValueError(f"meshes[{i}] must be 3D, got {m.dims}D")

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

    suffix = build_suffix(meta0)
    wrappers = [m._wrapper for m in meshes]

    func_name = f"intersection_curves_list_{suffix}"
    (paths_offsets, paths_data), points = getattr(_trueform.intersect, func_name)(
        wrappers, mode
    )
    return OffsetBlockedArray(paths_offsets, paths_data), points
