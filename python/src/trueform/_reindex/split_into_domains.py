"""
split_into_domains() function implementation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import Tuple, Union

import numpy as np

from .. import _trueform
from .._core import OffsetBlockedArray
from .._dispatch import InputMeta, build_suffix, ensure_mesh


def split_into_domains(
    mesh,
    domain_labels: Union[
        Tuple[np.ndarray, int],
        Tuple[np.ndarray, int, int],
    ],
):
    """
    Split a 3D polygon mesh into per-domain watertight submeshes.

    Each face contributes one emission per side bound to a real domain.
    Sides carrying the sentinel (open-fragment garbage or excluded outer
    shell) are dropped. Side-0 emissions reverse the stored winding so every
    output submesh has outward-of-domain normals.

    Parameters
    ----------
    mesh : tf.Mesh or (faces, points) tuple
        3D triangle (ngon=3) or dynamic (variable-sized faces via
        OffsetBlockedArray) polygon mesh.
    domain_labels : (labels, n_domains) or (labels, n_domains, outer_shell_label)
        Accepts either the 3-tuple returned by ``tf.domain_labels`` (the
        third element is ignored — ``outer_shell_label`` is not consumed by
        the split) or a bare ``(labels, n_domains)`` pair. ``labels`` must
        be shape ``(n_faces, 2)`` with dtype matching the input face indices.

    Returns
    -------
    components : list
        One submesh per domain, each watertight with outward-of-domain
        normals. For fixed-triangle meshes each entry is
        ``(faces_ndarray, points_ndarray)``. For dynamic meshes each entry
        is ``(OffsetBlockedArray, points_ndarray)``.
    comp_labels : np.ndarray
        Shape (n_components,). The domain id for each emitted submesh.
    """
    mesh = ensure_mesh(mesh, dims=3)
    if not mesh.is_dynamic and mesh.ngon != 3:
        raise ValueError(
            f"split_into_domains: mesh must have triangular faces or be "
            f"dynamic, got {mesh.ngon} vertices per face."
        )

    # outer_shell_label (when present) is ignored — unused by C++ side.
    labels_2d, n_domains = domain_labels[0], domain_labels[1]
    if not isinstance(labels_2d, np.ndarray) or labels_2d.ndim != 2 \
            or labels_2d.shape[1] != 2:
        raise ValueError(
            "split_into_domains: domain_labels[0] must be an ndarray of "
            "shape (n_faces, 2)"
        )
    n_faces = len(mesh.faces)
    if labels_2d.shape[0] != n_faces:
        raise ValueError(
            f"split_into_domains: domain_labels has {labels_2d.shape[0]} "
            f"rows but mesh has {n_faces} faces"
        )
    if labels_2d.dtype != mesh.faces.dtype:
        raise TypeError(
            f"split_into_domains: domain_labels dtype ({labels_2d.dtype}) "
            f"must match faces dtype ({mesh.faces.dtype})"
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
    cpp_func = getattr(_trueform.reindex, f"split_into_domains_{suffix}")

    faces_arg = mesh._wrapper.faces_array() if mesh.is_dynamic else mesh.faces
    components, comp_labels = cpp_func(
        faces_arg, mesh.points, labels_2d, int(n_domains),
    )

    if mesh.is_dynamic:
        wrapped = [
            (OffsetBlockedArray(offsets, data_arr), comp_points)
            for (offsets, data_arr), comp_points in components
        ]
        return wrapped, comp_labels
    return components, comp_labels
