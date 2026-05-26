"""
sidedness_relations() function implementation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import Tuple

import numpy as np

from .. import _trueform
from .._core import OffsetBlockedArray
from .._dispatch import InputMeta, build_suffix, ensure_mesh


def sidedness_relations(
    mesh,
    tag_labels: np.ndarray,
) -> Tuple[Tuple[OffsetBlockedArray, OffsetBlockedArray], Tuple[int, np.ndarray]]:
    """
    Per-component sidedness against each operand mesh at the arrangement cuts.

    For each manifold-edge-connected component of the input arrangement,
    emits a block of ``(operand_tag, sidedness)`` entries listing the
    operands the component contacts along a non-manifold (cut) edge and
    the sidedness of the component against that operand's oriented
    surface.

    Parameters
    ----------
    mesh : tf.Mesh or (faces, points) tuple
        3D triangle (ngon=3) or dynamic (variable-sized faces via
        OffsetBlockedArray) polygon mesh, typically the output of
        ``tf.mesh_arrangements``.
    tag_labels : np.ndarray
        Per-face source-mesh tag, 1D array with the same integer dtype
        as the mesh faces. Typically the second return of
        ``tf.mesh_arrangements``.

    Returns
    -------
    relations : (OffsetBlockedArray, OffsetBlockedArray)
        Two parallel offset-blocked arrays keyed per component, sharing
        the same offsets. The first carries operand tag ids; the second
        carries sidedness values (see ``tf.sidedness``). For component
        ``c``:

            tags  = relations[0][c]   # operand tags
            sides = relations[1][c]   # sidedness values

        Empty blocks indicate components that don't contact any other
        operand at a non-manifold edge.
    component_labels : (int, np.ndarray)
        Per-face manifold-edge connected-component info:
        ``(n_components, labels)`` — ``labels[face_id]`` is the component
        id of that face.
    """
    mesh = ensure_mesh(mesh, dims=3)
    if not mesh.is_dynamic and mesh.ngon != 3:
        raise ValueError(
            f"sidedness_relations: mesh must have triangular faces or be "
            f"dynamic, got {mesh.ngon} vertices per face."
        )

    if tag_labels.dtype != mesh.faces.dtype:
        raise TypeError(
            f"tag_labels dtype must match mesh.faces dtype "
            f"({mesh.faces.dtype}), got {tag_labels.dtype}."
        )
    if tag_labels.ndim != 1:
        raise ValueError(
            f"tag_labels must be 1D, got shape {tag_labels.shape}."
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
    cpp_func = getattr(_trueform.topology, f"sidedness_relations_{suffix}")

    (tags_wrap, sides_wrap), (n_components, labels) = cpp_func(
        mesh._wrapper, tag_labels
    )

    tags_oba = OffsetBlockedArray(tags_wrap.offsets_array(), tags_wrap.data_array())
    sides_oba = OffsetBlockedArray(sides_wrap.offsets_array(), sides_wrap.data_array())

    return (tags_oba, sides_oba), (int(n_components), labels)
