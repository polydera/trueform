"""
outer_shell() function implementation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from .. import _trueform
from .._core import OffsetBlockedArray
from .._dispatch import InputMeta, build_suffix
from .._spatial import Mesh

# Mirrors the tf::domain_config C++ enum values.
_IGNORE_OPEN_FRAGMENTS = 2


def outer_shell(mesh: Mesh, *, ignore_open_fragments: bool = False) -> Mesh:
    """
    Repair a mesh to its outer shell: the boundary of the union of
    everything it encloses.

    Splits the mesh at its self-intersection curves, labels the volumetric
    domains, and keeps only the faces bounding the unbounded outside,
    oriented outward. Internal structure — overlap membranes between
    interpenetrating parts, faces buried inside the solid, enclosed
    cavities — is removed. The result is free of self-intersections and
    suitable as a boolean or CsgGraph operand.

    Parameters
    ----------
    mesh : Mesh
        3D mesh to repair (triangle or dynamic).
    ignore_open_fragments : bool, default False
        Mask open fragments (faces in components carrying boundary edges)
        out of the region formation, so fins and damage do not partition
        the enclosed volume.

    Returns
    -------
    shell : Mesh
        The outer shell mesh, with the same face-index and point dtypes
        as the input.

    Examples
    --------
    >>> import trueform as tf
    >>> mesh = tf.Mesh(*tf.read_stl("self_intersecting.stl"))
    >>> shell = tf.outer_shell(mesh)
    """
    if not isinstance(mesh, Mesh):
        raise TypeError(
            f"mesh must be a Mesh object, got {type(mesh).__name__}. "
            f"Topology information is required for outer shell extraction."
        )

    if mesh.dims != 3:
        raise ValueError(
            f"outer_shell only supports 3D meshes, got mesh with {mesh.dims}D"
        )

    config = 0
    if ignore_open_fragments:
        config |= _IGNORE_OPEN_FRAGMENTS

    ngon = "dyn" if mesh.is_dynamic else str(mesh.ngon)
    meta = InputMeta(mesh.faces.dtype, mesh.dtype, ngon, 3)
    suffix = build_suffix(meta)

    func = getattr(_trueform.csg, f"outer_shell_{suffix}", None)
    if func is None:
        raise TypeError(f"unsupported mesh type combination for outer_shell: {suffix}")

    faces, points = func(mesh._wrapper, config)
    if mesh.is_dynamic:
        faces = OffsetBlockedArray(faces[0], faces[1])
    return Mesh(faces, points)
