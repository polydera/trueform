"""
has_self_intersections() function implementation

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from .. import _trueform
from .._spatial import Mesh
from .._dispatch import InputMeta, build_suffix


def has_self_intersections(mesh: Mesh) -> bool:
    """
    Check whether a mesh meets itself.

    True exactly when the self-intersection build would state a record:
    the discovery is run in exact arithmetic and stopped at the first
    contact, so a mesh that meets itself is answered without the rest of
    it being seen. Faces sharing a vertex or an edge are neighbours
    rather than contacts, so a bowtie or a fan does not make the verdict
    true on its own.

    Parameters
    ----------
    mesh : Mesh
        3D mesh to check (triangle or dynamic).

    Returns
    -------
    bool
        True if any two non-neighbouring faces of the mesh touch,
        False otherwise.

    Examples
    --------
    >>> import trueform as tf
    >>> mesh = tf.Mesh(*tf.make_sphere_mesh(1.0))
    >>> tf.has_self_intersections(mesh)
    False
    """
    if not isinstance(mesh, Mesh):
        raise TypeError(
            f"mesh must be a Mesh object, got {type(mesh).__name__}. "
            f"Topology information is required for self-intersection checks."
        )

    if mesh.dims != 3:
        raise ValueError(
            f"has_self_intersections only supports 3D meshes, got mesh with {mesh.dims}D"
        )

    ngon = 'dyn' if mesh.is_dynamic else str(mesh.ngon)
    meta = InputMeta(mesh.faces.dtype, mesh.dtype, ngon, 3)
    suffix = build_suffix(meta)

    func = getattr(_trueform.intersect, f"has_self_intersections_{suffix}")
    return func(mesh._wrapper)
