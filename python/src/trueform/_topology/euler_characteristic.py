"""
euler_characteristic() function implementation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from .. import _trueform
from .._dispatch import InputMeta, build_suffix, ensure_mesh


def euler_characteristic(mesh) -> int:
    """
    Compute the Euler characteristic of a 3D polygon mesh.

    The Euler characteristic is ``V - E + F``, where V is the number of
    vertices, E the number of unique edges, and F the number of faces.
    Each undirected edge is counted exactly once, so boundary and
    non-manifold edges count like interior ones: a closed sphere gives 2,
    a torus 0, a single triangle 1, and an open disk-like patch 1.

    Parameters
    ----------
    mesh : tf.Mesh or (faces, points) tuple
        3D triangle (ngon=3) or dynamic (variable-sized faces via
        OffsetBlockedArray) polygon mesh.

    Returns
    -------
    int
        The Euler characteristic ``V - E + F``.

    Examples
    --------
    >>> import trueform as tf
    >>> mesh = tf.Mesh(*tf.make_sphere_mesh(1.0))
    >>> tf.euler_characteristic(mesh)
    2
    """
    mesh = ensure_mesh(mesh, dims=3)
    if not mesh.is_dynamic and mesh.ngon != 3:
        raise ValueError(
            f"euler_characteristic: mesh must have triangular faces or be "
            f"dynamic, got {mesh.ngon} vertices per face."
        )

    ngon = "dyn" if mesh.is_dynamic else "3"
    suffix = build_suffix(
        InputMeta(
            index_dtype=mesh.faces.dtype,
            real_dtype=mesh.points.dtype,
            ngon=ngon,
            dims=3,
        )
    )
    cpp_func = getattr(_trueform.topology, f"euler_characteristic_{suffix}")
    return int(cpp_func(mesh._wrapper))
