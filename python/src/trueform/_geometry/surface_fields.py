"""
Fields over a surface: geodesic distances and the Dirichlet field

Copyright (c) 2026 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple, Union

import numpy as np

from .. import _trueform
from .._core import OffsetBlockedArray
from .._dispatch import ensure_mesh, extract_meta, build_suffix
from .._spatial import Mesh

MeshLike = Union[Mesh, Tuple[np.ndarray, np.ndarray], Tuple[OffsetBlockedArray, np.ndarray]]


_ORDER_MAP = {"harmonic": 0, "biharmonic": 1}


class DirichletFieldStatus(IntEnum):
    """What a :func:`dirichlet_field` solve answered (C++
    ``tf::dirichlet_field_status``).

    ``REFUSED_OPERATOR`` is a corner weight that is not finite, or a point
    whose lumped mass is not positive — one no face has as a corner, or one
    whose every face is degenerate; ``REFUSED_UNANCHORED`` a connected run of
    free points reaching no fixed one; ``REFUSED_INDEX`` an index type too
    narrow to name the factor; ``REFUSED_INDEFINITE`` a pivot that is not
    positive; ``REFUSED_RESIDUAL`` an answer whose true residual misses its
    acceptance.
    """
    SOLVED = 0
    REFUSED_OPERATOR = 1
    REFUSED_UNANCHORED = 2
    REFUSED_INDEX = 3
    REFUSED_INDEFINITE = 4
    REFUSED_RESIDUAL = 5


@dataclass(frozen=True)
class DirichletField:
    """What :func:`dirichlet_field` solved (C++ ``tf::dirichlet_field_result``).

    Attributes
    ----------
    field : np.ndarray
        One value per point, in the mesh's dtype. The fixed values stand at
        the fixed points whatever happened; a solved field states its answer
        everywhere else, and a refused one states zero there.
    status : DirichletFieldStatus
        What the system answered.
    """
    field: np.ndarray
    status: DirichletFieldStatus


def _point_ids(ids, mesh: Mesh, name: str, distinct: bool) -> np.ndarray:
    ids = np.asarray(ids)
    if ids.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {ids.shape}")
    if ids.size and not np.issubdtype(ids.dtype, np.integer):
        raise TypeError(f"{name} must hold integer point ids, got {ids.dtype}")
    n = mesh.number_of_points
    if ids.size and (ids.min() < 0 or ids.max() >= n):
        raise IndexError(f"{name}: point id out of range [0, {n})")
    ids = np.ascontiguousarray(ids, dtype=mesh.faces.dtype)
    if distinct and len(np.unique(ids)) != len(ids):
        raise ValueError(f"{name}: point ids must be distinct")
    return ids


def geodesic_distances(
    data: MeshLike,
    seeds,
    radius: Optional[float] = None,
) -> np.ndarray:
    """
    The geodesic distance from the nearest seed to every point.

    A label-correcting front with the two-neighbour triangle-unfolding
    update: a seed's ring is reached at its Euclidean distance, and the front
    runs from there. On a planar mesh it is the straight line; on a curved
    one it beats the shortest edge path. Distances are measured in the stored
    coordinates; a mesh's transformation does not enter them.

    Parameters
    ----------
    data : Mesh or tuple
        A 3D mesh of triangles or dynamic faces, or its (faces, points).
    seeds : array-like of int
        The point ids the distance is measured from.
    radius : float, optional
        The reach of the front. If None, the front is unbounded.

    Returns
    -------
    np.ndarray of shape (num_points,), the mesh's dtype
        Zero at a seed, infinity where no front reaches — another piece of
        the mesh, or past ``radius`` — and at most ``radius`` otherwise.

    Examples
    --------
    >>> import trueform as tf
    >>> faces, points = tf.make_sphere_mesh(1.0, 32, 32)
    >>> d = tf.geodesic_distances(tf.Mesh(faces, points), [0])
    """
    mesh = ensure_mesh(data, dims=3)
    if not mesh.is_dynamic and mesh.ngon != 3:
        raise ValueError(
            f"mesh must have triangular faces or be dynamic, got {mesh.ngon} vertices per face")
    seeds = _point_ids(seeds, mesh, "seeds", distinct=False)
    radius = np.inf if radius is None else float(radius)
    if np.isnan(radius) or radius < 0:
        raise ValueError("radius must be non-negative and not NaN")
    suffix = build_suffix(extract_meta(mesh))
    return getattr(_trueform.geometry, f"geodesic_distances_{suffix}")(
        mesh._wrapper, seeds, radius)


def dirichlet_field(
    data: MeshLike,
    fixed_ids,
    fixed_values,
    order: str,
    corner_weights: Optional[np.ndarray] = None,
) -> DirichletField:
    """
    The field held at ``fixed_ids`` and harmonic or biharmonic elsewhere.

    ``"harmonic"`` solves ``L x = 0`` and ``"biharmonic"`` ``L M^-1 L x = 0``
    at every free point, where ``L`` is the Laplacian the corner weights state
    and ``M`` the barycentric lumped mass. The free rows are solved by sparse
    Cholesky, refined once and accepted against their true residual. The
    system is definite where every weight is positive and every connected run
    of free points reaches a fixed one; the result's ``status`` names what
    refused.

    Parameters
    ----------
    data : Mesh or tuple
        A 3D triangle mesh, or its (faces, points). Read in the stored
        coordinates.
    fixed_ids : array-like of int
        The distinct point ids the field is held at.
    fixed_values : array-like of float
        The finite value at each of them.
    order : str
        ``"harmonic"`` or ``"biharmonic"``.
    corner_weights : np.ndarray, optional
        ``3 * num_faces`` weights; ``corner_weights[3 f + c]`` weighs
        the edge of face ``f`` opposite its corner ``c``, summed over the
        faces an edge borders. None states the cotangent weights.

    Returns
    -------
    DirichletField

    Examples
    --------
    >>> import trueform as tf
    >>> mesh = tf.Mesh(faces, points)
    >>> result = tf.dirichlet_field(mesh, [0, 1], [0.0, 1.0], "harmonic")
    >>> result.status, result.field
    """
    mesh = ensure_mesh(data, dims=3)
    if mesh.is_dynamic or mesh.ngon != 3:
        raise ValueError("dirichlet_field is stated over triangle meshes")
    fixed_ids = _point_ids(fixed_ids, mesh, "fixed_ids", distinct=True)
    fixed_values = np.ascontiguousarray(fixed_values, dtype=np.float64)
    if fixed_values.ndim != 1 or len(fixed_values) != len(fixed_ids):
        raise ValueError("dirichlet_field: one fixed value per fixed id")
    if not np.all(np.isfinite(fixed_values)):
        raise ValueError("dirichlet_field: fixed values must be finite")
    if order not in _ORDER_MAP:
        raise ValueError(f"order must be 'harmonic' or 'biharmonic', got '{order}'")
    faces = mesh.faces
    suffix = build_suffix(extract_meta(mesh))
    native = getattr(_trueform.geometry, f"dirichlet_field_{suffix}")
    if corner_weights is None:
        field, status = native(mesh._wrapper, fixed_ids, fixed_values,
                               _ORDER_MAP[order])
    else:
        corner_weights = np.ascontiguousarray(corner_weights, dtype=np.float64)
        if corner_weights.ndim != 1 or len(corner_weights) != 3 * len(faces):
            raise ValueError(
                "dirichlet_field: corner_weights must hold three weights per face")
        field, status = native(mesh._wrapper, corner_weights, fixed_ids,
                               fixed_values, _ORDER_MAP[order])
    return DirichletField(field=field, status=DirichletFieldStatus(status))
