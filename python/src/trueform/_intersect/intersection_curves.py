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


def intersection_curves(
    meshes_or_mesh0: Union[Mesh, List[Mesh]],
    mesh1: Mesh = None,
    *,
    mode: str = "primitives",
    tolerance: float = 0.0,
    within: bool = False
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
    mode : str, default "primitives"
        The classifier the run states its contacts with. "primitives"
        classifies shared edges/vertices and coplanar contacts; "sos"
        perturbs every contact into a crossing and cannot state shared or
        coplanar geometry. Crossings between contours are resolved
        whenever the operands can make such a pair.
    tolerance : float, default 0.0
        World-coordinate distance an input vertex may move to reach the lattice
        (0 = exact).
    within : bool, default False
        Also intersect each mesh with itself, so self-crossings resolve in
        the arrangement the curves are read from. The emitted curves stay
        the cross-mesh seams — an operand's own seam is
        `self_intersection_curves`' product.

    Returns
    -------
    paths : OffsetBlockedArray
        Paths as indices into the points array. Each path is one curve.
    points : np.ndarray
        Curve point coordinates with shape (N, 3).
    """

    if mode not in _MODE_MAP:
        raise ValueError(f"mode must be 'sos' or 'primitives', got '{mode}'")
    m = _MODE_MAP[mode] | (4 if within else 0)

    if isinstance(meshes_or_mesh0, (list, tuple)):
        return _intersection_curves_list(meshes_or_mesh0, mode=m, tolerance=tolerance)
    else:
        if mesh1 is None:
            raise ValueError(
                "intersection_curves requires either two meshes or a list of meshes"
            )
        return _intersection_curves_pair(meshes_or_mesh0, mesh1, mode=m, tolerance=tolerance)


def _intersection_curves_pair(mesh0: Mesh, mesh1: Mesh, *, mode: int, tolerance: float) -> Tuple[OffsetBlockedArray, np.ndarray]:
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
        mesh0._wrapper, mesh1._wrapper, mode, tolerance
    )
    return OffsetBlockedArray(paths_offsets, paths_data), points


def _intersection_curves_list(meshes: list, *, mode: int, tolerance: float) -> Tuple[OffsetBlockedArray, np.ndarray]:
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
        wrappers, mode, tolerance
    )
    return OffsetBlockedArray(paths_offsets, paths_data), points
