"""
Index maps returned by CsgGraph queries.

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""
from dataclasses import dataclass

from .._core import OffsetBlockedArray


@dataclass(frozen=True)
class MeshIndexMap:
    """
    Output <-> input identity maps for a ``CsgGraph.mesh`` result.

    Attributes
    ----------
    point_tag_labels : np.ndarray
        Output point -> input mesh tag; created points carry ``n_tags``.
    point_labels : np.ndarray
        Output point -> input point id within its mesh; created points
        carry ``n_output_points``.
    face_tag_labels : np.ndarray
        Output face -> input mesh tag.
    face_labels : np.ndarray
        Output face -> input face id (origin face for cut faces).
    point_f : OffsetBlockedArray
        Forward map: ``point_f[tag][input id]`` -> output point id
        (dropped inputs carry ``n_output_points``).
    n_original_points, n_original_faces : int
        Output entries below these are kept originals; at/above, created.
    n_tags : int
        Number of input meshes; the tag-axis end sentinel.
    n_output_points : int
        Total output points; the point-id-axis end sentinel.
    """

    point_tag_labels: object
    point_labels: object
    face_tag_labels: object
    face_labels: object
    point_f: OffsetBlockedArray
    n_original_points: int
    n_original_faces: int
    n_tags: int
    n_output_points: int


@dataclass(frozen=True)
class DomainsIndexMap:
    """
    Per-cell identity maps for a ``CsgGraph.domains`` result; every
    block array is parallel to the cell list.

    Attributes
    ----------
    face_tag_blocks : OffsetBlockedArray
        ``face_tag_blocks[k][j]`` -> input form of cell k's face j.
    face_blocks : OffsetBlockedArray
        ``face_blocks[k][j]`` -> original face id within that form.
    point_tag_blocks : OffsetBlockedArray
        ``point_tag_blocks[k][j]`` -> input form of cell k's point j;
        created points carry ``n_tags``.
    point_blocks : OffsetBlockedArray
        ``point_blocks[k][j]`` -> input point id within that form;
        created points carry ``n_output_points``.
    n_original_points : int
        Global arrangement points below this are kept originals.
    n_tags : int
        Number of input forms; the tag-axis end sentinel.
    n_output_points : int
        Total global arrangement points; the point-id-axis end sentinel.
    inclusion : np.ndarray
        ``(n_cells, n_tags)`` bool matrix; ``inclusion[k, i]`` is True iff
        cell k lies inside form i (for a sheet: behind its normal).
        Boolean masks over it select cells post-extraction:
        ``inclusion[:, 0] & ~inclusion[:, 1]``.
    """

    face_tag_blocks: OffsetBlockedArray
    face_blocks: OffsetBlockedArray
    point_tag_blocks: OffsetBlockedArray
    point_blocks: OffsetBlockedArray
    n_original_points: int
    n_tags: int
    n_output_points: int
    inclusion: object
