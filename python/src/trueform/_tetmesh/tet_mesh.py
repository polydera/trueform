"""
tet_mesh() implementation

Copyright (c) 2026 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np

from .. import _trueform
from .._core import OffsetBlockedArray
from .._dispatch import InputMeta, build_suffix
from .._spatial import Mesh


class TetMeshStatus(IntEnum):
    """What :func:`tet_mesh` made of its input (C++ ``tf::tet_mesh_status``)."""
    MESHED = 0
    REFUSED = 1
    UNWOUND = 2
    RANK_DEFICIENT = 3
    CAPACITY = 4
    OPEN = 5


@dataclass(frozen=True)
class TetMeshReport:
    """The status of a :func:`tet_mesh` call and what it counted
    (C++ ``tf::tet_mesh_report``).

    ``census`` is the constraint recovery's census as a dict, its face
    routes nested under ``census["faces"]``.
    """
    status: TetMeshStatus
    refused_segments: int
    refused_depth: int
    refused_budget: int
    refused_faces: int
    rung: int
    drains: int
    pairs_named: int
    pairs_cut: int
    triangles_cut: int
    dropped_degenerate: int
    stitched_triangles: int
    failed_triangles: int
    pairs_unresolved: int
    input_open_edges: int
    recovered_open_edges: int
    storage_bits: int
    input_bits: int
    frame_bits: int
    deep_sites: int
    deepest: int
    corner_bits_needed: int
    min_angle: float
    flat: int
    inverted: int
    negative_cells: int
    census: dict


@dataclass(frozen=True)
class TetMeshIndexMap:
    """Where the input surface stands in its tet mesh
    (C++ ``tf::tet_mesh_index_map``).

    Attributes
    ----------
    facet_offsets, facet_triangles : np.ndarray
        Per facet, the boundary triangles covering it: rows
        ``facet_offsets[k]:facet_offsets[k + 1]`` of the (N, 3) corners,
        indexed into the tet mesh's points and wound as the facet is.
    face_facets : OffsetBlockedArray
        Per input face, the facets it lies in, ascending; one unless a
        crossing cut the face.
    """
    facet_offsets: np.ndarray
    facet_triangles: np.ndarray
    face_facets: OffsetBlockedArray


def tet_mesh(
    mesh: Mesh,
    *,
    split_constraints: bool = True,
    tolerance: float = 0.0,
    site_budget_ratio: Optional[int] = None,
    return_report: bool = False,
    return_index_map: bool = False,
):
    """
    Tetrahedralize the solid a closed triangle surface winds.

    Crossing faces are cut at their canonical intersections, every face and
    edge is recovered exactly as a constraint, and the cells the surface
    winds positively are kept. The mesh's transformation places it: the
    points come back in world space, in float64 whatever the mesh's dtype,
    because they are lattice positions finer than a float32 holds.

    Parameters
    ----------
    mesh : Mesh
        A 3D triangle mesh.
    split_constraints : bool, default True
        Cut the crossings the constraint recovery finds, where it finds
        them. Without it the surface is meshed as given: no crossing is cut,
        and the first one the recovery names refuses.
    tolerance : float, default 0.0
        Plane pooling, in the mesh's coordinates (0 = exact): near-coplanar
        faces pool onto one plane at the entrance, and it is the band of
        every cut round.
    site_budget_ratio : int, optional
        The sites a run may hold per site its entrance states; a recovery
        that would pass it refuses. Default: the library's.
    return_report : bool, default False
        Return the :class:`TetMeshReport` in the status's place: the
        counts that name a refusal's reason, the lattice that ran, and the
        diagnostics no status needs, measured only when asked.
    return_index_map : bool, default False
        Also return the :class:`TetMeshIndexMap`.

    Returns
    -------
    tetrahedra : np.ndarray
        (T, 4) corners into ``points``.
    points : np.ndarray
        (P, 3) float64.
    status : TetMeshStatus
        Or the :class:`TetMeshReport` when ``return_report`` is set.
    index_map : TetMeshIndexMap
        Only when ``return_index_map`` is set.

    Examples
    --------
    >>> import trueform as tf
    >>> tets, points, status = tf.tet_mesh(tf.Mesh(faces, points))
    >>> status == tf.TetMeshStatus.MESHED
    """
    if not isinstance(mesh, Mesh):
        raise TypeError(f"mesh must be Mesh, got {type(mesh).__name__}")
    if mesh.dims != 3:
        raise ValueError(f"tet_mesh only supports 3D meshes, got {mesh.dims}D")
    if mesh.is_dynamic or mesh.ngon != 3:
        raise ValueError("tet_mesh takes a triangle mesh; call triangulated() first")
    if site_budget_ratio is not None and (
            isinstance(site_budget_ratio, bool)
            or not isinstance(site_budget_ratio, (int, np.integer))):
        raise TypeError(
            "site_budget_ratio must be an int, got "
            f"{type(site_budget_ratio).__name__}")
    suffix = build_suffix(InputMeta(mesh.faces.dtype, mesh.dtype, '3', 3))
    raw = getattr(_trueform.tetmesh, f"tet_mesh_{suffix}")(
        mesh._wrapper, bool(split_constraints), float(tolerance),
        None if site_budget_ratio is None else int(site_budget_ratio),
        bool(return_report), bool(return_index_map))
    tetrahedra, points = raw[0], raw[1]
    if return_report:
        fields = dict(raw[2])
        fields["status"] = TetMeshStatus(fields["status"])
        answer = TetMeshReport(**fields)
    else:
        answer = TetMeshStatus(raw[2])
    if not return_index_map:
        return tetrahedra, points, answer
    facet_offsets, facet_triangles, face_offsets, face_data = raw[3]
    return tetrahedra, points, answer, TetMeshIndexMap(
        facet_offsets=facet_offsets, facet_triangles=facet_triangles,
        face_facets=OffsetBlockedArray(face_offsets, face_data))
