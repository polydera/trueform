"""
domain_labels() function implementation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import Tuple

import numpy as np

from .. import _trueform
from .._dispatch import InputMeta, build_suffix, ensure_mesh

# Mirrors the tf::domain_config C++ enum values.
_EXCLUDE_OUTER_SHELL = 1
_IGNORE_OPEN_FRAGMENTS = 2


def domain_labels(
    mesh,
    *,
    ignore_open_fragments: bool = False,
    exclude_outer_shell: bool = False,
) -> Tuple[np.ndarray, int, int]:
    """
    Compute per-face per-side volumetric domain labels for a 3D polygon mesh.

    A non-manifold surface mesh bounds multiple 3D regions ("domains"). This
    function partitions space into volumetric components and returns one
    label per face per side.

    Parameters
    ----------
    mesh : tf.Mesh or (faces, points) tuple
        3D triangle (ngon=3) or dynamic (variable-sized faces via
        OffsetBlockedArray) polygon mesh.
    ignore_open_fragments : bool, default False
        Drop face-sides bounding open fragments (faces in MEL components
        carrying boundary edges) by parking them at the sentinel label
        ``n_domains``.
    exclude_outer_shell : bool, default False
        Fold the unbounded universe domain into the same sentinel, so only
        bounded interior domains receive ids.

    Returns
    -------
    labels : np.ndarray
        Shape (n_faces, 2), dtype matching the input face indices.
        ``labels[f, 0]`` = domain id on the stored-normal side of face ``f``;
        ``labels[f, 1]`` = domain id on the reversed side.
    n_domains : int
        Number of distinct bounded domains.
    outer_shell_label : int
        Label carried by outer-shell sides. Equals ``n_domains`` under
        ``exclude_outer_shell``; otherwise no face-side carries this label.
    """
    mesh = ensure_mesh(mesh, dims=3)
    if not mesh.is_dynamic and mesh.ngon != 3:
        raise ValueError(
            f"domain_labels: mesh must have triangular faces or be dynamic, "
            f"got {mesh.ngon} vertices per face."
        )

    config = 0
    if exclude_outer_shell:
        config |= _EXCLUDE_OUTER_SHELL
    if ignore_open_fragments:
        config |= _IGNORE_OPEN_FRAGMENTS

    ngon = "dyn" if mesh.is_dynamic else "3"
    suffix = build_suffix(
        InputMeta(
            index_dtype=mesh.faces.dtype,
            real_dtype=mesh.points.dtype,
            ngon=ngon,
            dims=3,
        )
    )
    cpp_func = getattr(_trueform.topology, f"domain_labels_{suffix}")

    flat_labels, n_domains, outer_shell_label = cpp_func(mesh._wrapper, config)
    return flat_labels.reshape(-1, 2), int(n_domains), int(outer_shell_label)
