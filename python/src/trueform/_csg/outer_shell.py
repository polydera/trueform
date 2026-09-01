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


def outer_shell(mesh: Mesh) -> Mesh:
    """
    Repair a mesh to its outer shell: the boundary of the union of
    everything it encloses.

    The mesh is read through its own CSG graph — the self arrangement plus
    the classification tier — and only the faces bounding the unbounded
    outside are kept, oriented outward. Internal structure — overlap
    membranes between interpenetrating parts, faces buried inside the
    solid, enclosed cavities — has the same domain on both sides and so
    never reaches the boundary. Open fragments (fins, damage) are
    self-merged by the arrangement, so they bound no volume and cannot
    survive into the shell.

    The extraction is structural: no winding bits, no expression, no
    options. An uncut vertex reaches the output with its input coordinate
    untouched. The result is free of self-intersections and suitable as a
    boolean or CsgGraph operand.

    Parameters
    ----------
    mesh : Mesh
        3D mesh to repair (triangle or dynamic).

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

    ngon = "dyn" if mesh.is_dynamic else str(mesh.ngon)
    meta = InputMeta(mesh.faces.dtype, mesh.dtype, ngon, 3)
    suffix = build_suffix(meta)

    func = getattr(_trueform.csg, f"outer_shell_{suffix}", None)
    if func is None:
        raise TypeError(f"unsupported mesh type combination for outer_shell: {suffix}")

    faces, points = func(mesh._wrapper)
    if mesh.is_dynamic:
        faces = OffsetBlockedArray(faces[0], faces[1])
    return Mesh(faces, points)
