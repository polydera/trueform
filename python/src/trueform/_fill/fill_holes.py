"""
fill_holes() and filled_mesh() implementation

Copyright (c) 2026 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple, Union

import numpy as np

from .. import _trueform
from .._core import OffsetBlockedArray
from .._dispatch import InputMeta, build_suffix
from .._spatial import Mesh


class HoleFillStatus(IntEnum):
    """What one hole group's fill produced (C++ ``tf::hole_fill_status``)."""
    FILLED = 0
    REFUSED_OPEN = 1
    REFUSED_INVALID_RIM = 2
    REFUSED_HOST_FACE = 3
    REFUSED_PROTECTION_DEPTH = 4
    REFUSED_TABLE = 5
    REFUSED_RESOURCE = 6
    REFUSED_RIM_SPLIT = 7


class HoleChartStatus(IntEnum):
    """How the chart tier read a group's hole (C++ ``tf::hole_chart_status``).

    ``ADMITTED`` and ``RIM_PLANE`` hold the patch: the first continues the
    host, the second is the flat lid of an exactly planar rim. Every other
    value is the reason the chart refused and handed the rim to the table.
    """
    NOT_ATTEMPTED = 0
    ADMITTED = 1
    RIM_PLANE = 2
    NO_AREA = 3
    STEEP = 4
    FOLDED = 5
    COLLAR_ARITY = 6
    UNCOVERED = 7
    UNSAMPLED = 8
    UNSOLVED = 9
    FORBIDDEN_CHORD = 10
    RESOURCE = 11
    OUT_OF_LATTICE = 12


class HoleRefineStatus(IntEnum):
    """How a group's patch left refinement (C++ ``tf::hole_refine_status``)."""
    NOT_ATTEMPTED = 0
    FLOOR_MET = 1
    STALLED = 2
    BUDGET_EXHAUSTED = 3


class HoleFairStatus(IntEnum):
    """How a group's patch left fairing (C++ ``tf::hole_fair_status``)."""
    NOT_ATTEMPTED = 0
    FAIRED = 1
    FAIRING_OFF = 2
    FAIRING_REFUSED = 3


@dataclass(frozen=True)
class HoleFill:
    """What :func:`fill_holes` made of a mesh's rims (C++ ``tf::hole_fill_result``).

    One rim is one group; block ``g`` of every jagged lane and entry ``g`` of
    every flat lane belong to group ``g``. A corner below the mesh's point
    count is one of its vertices; at or above it, it names ``minted_points``
    read in order. A refused group holds an empty patch and names the rim edge it
    refused on in ``offending``.

    Attributes
    ----------
    group_offsets : np.ndarray
        Per group, the run of rims it fills, shape (G + 1,).
    triangle_offsets, triangles : np.ndarray
        Per group, its patch triangles: rows
        ``triangle_offsets[g]:triangle_offsets[g + 1]`` of the (T, 3) corners.
    minted_offsets, minted_points : np.ndarray
        Per group, the points it minted, rows of the (M, 3) points.
    status, charted, refined, faired : np.ndarray
        Per group, uint8 values of :class:`HoleFillStatus`,
        :class:`HoleChartStatus`, :class:`HoleRefineStatus` and
        :class:`HoleFairStatus`.
    seam_max_angle : np.ndarray
        Per group, the largest angle across its seam in radians, -1 where
        not computed.
    offending : np.ndarray
        Per group, the (rim, edge) its status speaks of, (-1, -1) where none.
    splits : np.ndarray
        The original edges the fill split, shape (S, 4) of
        (face, v0, v1, point), in canonical order.
    split_parameters : np.ndarray
        Per split, its parameter from v0 as a numerator over 64, uint8.
    plan_faces : np.ndarray
        The carrying faces those splits reached, ascending.
    plan_offsets, plan_triangles : np.ndarray
        Per plan face, the rows of the (P, 3) triangles it becomes.
    """
    group_offsets: np.ndarray
    triangle_offsets: np.ndarray
    triangles: np.ndarray
    minted_offsets: np.ndarray
    minted_points: np.ndarray
    status: np.ndarray
    charted: np.ndarray
    refined: np.ndarray
    faired: np.ndarray
    seam_max_angle: np.ndarray
    offending: np.ndarray
    splits: np.ndarray
    split_parameters: np.ndarray
    plan_faces: np.ndarray
    plan_offsets: np.ndarray
    plan_triangles: np.ndarray

    def __len__(self) -> int:
        return len(self.status)


def _suffix(mesh: Mesh) -> str:
    if not isinstance(mesh, Mesh):
        raise TypeError(f"mesh must be Mesh, got {type(mesh).__name__}")
    if mesh.dims != 3:
        raise ValueError(f"fill_holes only supports 3D meshes, got {mesh.dims}D")
    if not mesh.is_dynamic and mesh.ngon != 3:
        raise ValueError(
            f"mesh must have triangular faces or be dynamic, got {mesh.ngon} vertices per face.")
    ngon = 'dyn' if mesh.is_dynamic else str(mesh.ngon)
    return build_suffix(InputMeta(mesh.faces.dtype, mesh.dtype, ngon, mesh.dims))


def fill_holes(
    mesh: Mesh,
    rims: Tuple[OffsetBlockedArray, OffsetBlockedArray, np.ndarray],
    min_quality: float = 0.3,
    fairing: bool = True,
    split_rim: bool = True,
) -> HoleFill:
    """
    Fill a mesh's boundary rims with patches of its own kind.

    One rim is one group, handed down a ladder of tiers: exhaustive
    enumeration for a rim of at most five vertices; then the hole read as a
    height field over the plane its rim spans, continuing the surface around
    it — or, where that chart refuses and the rim lies exactly in one plane,
    the flat lid in that plane; and otherwise the tetrahedralization of its
    sites, over which a two-stage table states the patch of least
    bottleneck angle and, under it, least area. A carrying face must be a
    triangle. The result is replayed into a mesh by :func:`filled_mesh`.

    Parameters
    ----------
    mesh : Mesh
        The mesh the rims are read against: 3D, triangles or dynamic faces.
    rims : tuple
        The rims to fill, as :func:`trueform.boundary_rims` returns them.
    min_quality : float, default 0.3
        The quality floor a coarse patch is refined toward; zero asks for no
        refinement. A chart patch samples at its own floor.
    fairing : bool, default True
        Whether a refined coarse patch is faired against its host.
    split_rim : bool, default True
        Whether a tier may split the host's original rim edges. Splits are
        published in the result and replayed by :func:`filled_mesh`, so the
        stitched mesh stays watertight; without them no original edge is
        touched, and a rim the table could only fill by a split refuses as
        ``REFUSED_RIM_SPLIT``.

    Returns
    -------
    HoleFill
        One group per rim.

    Examples
    --------
    >>> import trueform as tf
    >>> mesh = tf.Mesh(faces, points)
    >>> fill = tf.fill_holes(mesh, tf.boundary_rims(mesh))
    >>> faces, points = tf.filled_mesh(mesh, fill)
    """
    suffix = _suffix(mesh)
    if not isinstance(rims, tuple) or len(rims) != 3:
        raise TypeError("rims must be the (vertices, faces, closed) tuple boundary_rims returns")
    vertices, faces, closed = rims
    if not isinstance(vertices, OffsetBlockedArray) or not isinstance(faces, OffsetBlockedArray):
        raise TypeError("rims must hold OffsetBlockedArray vertices and faces")
    if vertices.dtype != mesh.faces.dtype or faces.dtype != mesh.faces.dtype:
        raise TypeError(
            f"rim index dtype ({vertices.dtype}) must match mesh face dtype ({mesh.faces.dtype})")
    closed = np.ascontiguousarray(np.asarray(closed, dtype=np.int8))
    if closed.ndim != 1 or len(closed) != len(vertices) or len(faces) != len(vertices):
        raise ValueError("rims must state one vertex block, face block and closed flag per rim")
    lanes = [np.ascontiguousarray(a) for a in (vertices.offsets, vertices.data,
                                               faces.offsets, faces.data)]
    (group_offsets, (triangle_offsets, triangles), (minted_offsets, minted_points),
     status, charted, refined, faired, seam_max_angle, offending, splits,
     split_parameters, plan_faces, (plan_offsets, plan_triangles)) = getattr(
        _trueform.fill, f"fill_holes_{suffix}")(
        mesh._wrapper, *lanes, closed,
        float(min_quality), bool(fairing), bool(split_rim))
    return HoleFill(
        group_offsets=group_offsets, triangle_offsets=triangle_offsets,
        triangles=triangles, minted_offsets=minted_offsets,
        minted_points=minted_points, status=status, charted=charted,
        refined=refined, faired=faired, seam_max_angle=seam_max_angle,
        offending=offending, splits=splits, split_parameters=split_parameters,
        plan_faces=plan_faces, plan_offsets=plan_offsets,
        plan_triangles=plan_triangles)


def filled_mesh(
    mesh: Mesh, fill: HoleFill
) -> Tuple[Union[np.ndarray, OffsetBlockedArray], np.ndarray]:
    """
    The mesh a fill states: the input, with what it changed.

    A face no split reached is copied at its own arity; a carrying face a
    split reached is replaced by its plan; the patches follow. The points are
    the input's own, then the minted points. A triangle mesh comes back as
    triangles, any other as dynamic faces.

    Parameters
    ----------
    mesh : Mesh
        The mesh the fill was stated against.
    fill : HoleFill
        The result of :func:`fill_holes` on that mesh.

    Returns
    -------
    faces : np.ndarray or OffsetBlockedArray
    points : np.ndarray
    """
    suffix = _suffix(mesh)
    if not isinstance(fill, HoleFill):
        raise TypeError(f"fill must be HoleFill, got {type(fill).__name__}")
    faces, points = getattr(_trueform.fill, f"filled_mesh_{suffix}")(
        mesh._wrapper, fill.triangle_offsets, fill.triangles,
        fill.minted_offsets, fill.minted_points, fill.plan_faces,
        fill.plan_offsets, fill.plan_triangles)
    if mesh.is_dynamic:
        faces = OffsetBlockedArray(faces[0], faces[1])
    return faces, points
