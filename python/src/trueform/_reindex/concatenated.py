"""
Concatenate multiple meshes or edge meshes into a single geometry.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import List, Tuple, Union
import numpy as np
from .._spatial import Mesh, EdgeMesh
from .._core import OffsetBlockedArray


def _apply_transformation(points: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    """Apply transformation matrix to points array, or return unchanged if None."""
    if transformation is None:
        return points
    dims = points.shape[1]
    ones = np.ones((len(points), 1), dtype=points.dtype)
    homogeneous = np.hstack([points, ones])
    transformed = (transformation @ homogeneous.T).T[:, :dims]
    return transformed.astype(points.dtype)


def concatenated(
    data: Union[List[Tuple[np.ndarray, np.ndarray]],
                List[Mesh], List[EdgeMesh]]
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[OffsetBlockedArray, np.ndarray]]:
    """
    Concatenate multiple meshes or edge meshes into a single geometry.

    Takes a list of geometric data and merges them into a single unified structure,
    automatically handling index offsetting to maintain referential integrity.
    If Mesh or EdgeMesh objects have transformations set, they are applied to the
    points before concatenation.

    Parameters
    ----------
    data : List[Tuple[np.ndarray, np.ndarray]] | List[Mesh] | List[EdgeMesh]
        List of geometries to concatenate. Can be:
        - List of (indices, points) tuples (indices can be ndarray or OffsetBlockedArray)
        - List of Mesh objects (including dynamic meshes)
        - List of EdgeMesh objects

        All geometries must have same dims (point dimensions): all 2D or all 3D.
        Dtypes can be mixed - numpy will handle type promotion automatically.
        Transformations on Mesh/EdgeMesh objects are applied to points.

        Mixing is allowed:
        - Fixed-size with different V (e.g., triangles + quads) → dynamic output
        - Fixed-size with dynamic → dynamic output

    Returns
    -------
    Tuple[np.ndarray, np.ndarray] or Tuple[OffsetBlockedArray, np.ndarray]
        (concatenated_indices, concatenated_points)
        - concatenated_indices: Combined indices with offsets applied.
          Returns OffsetBlockedArray if any input is dynamic or if V values differ.
          Returns np.ndarray only if all inputs are fixed-size with same V.
        - concatenated_points: Combined point coordinates

    Raises
    ------
    ValueError
        If input list is empty, or if dims don't match
    TypeError
        If input types are mixed (e.g., Mesh and EdgeMesh together)

    Examples
    --------
    >>> # Concatenate two triangle meshes
    >>> mesh1 = tf.Mesh(
    ...     np.array([[0, 1, 2]], dtype=np.int32),
    ...     np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    ... )
    >>> mesh2 = tf.Mesh(
    ...     np.array([[0, 1, 2]], dtype=np.int32),
    ...     np.array([[5, 5, 5], [6, 5, 5], [5, 6, 5]], dtype=np.float32)
    ... )
    >>> faces, points = tf.concatenated([mesh1, mesh2])
    >>> # faces: [[0, 1, 2], [3, 4, 5]]
    >>> # points: [[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5], [6, 5, 5], [5, 6, 5]]

    >>> # Concatenate edge meshes
    >>> edges1 = tf.EdgeMesh(
    ...     np.array([[0, 1]], dtype=np.int32),
    ...     np.array([[0, 0], [1, 0]], dtype=np.float32)
    ... )
    >>> edges2 = tf.EdgeMesh(
    ...     np.array([[0, 1]], dtype=np.int32),
    ...     np.array([[2, 2], [3, 2]], dtype=np.float32)
    ... )
    >>> edges, points = tf.concatenated([edges1, edges2])
    >>> # edges: [[0, 1], [2, 3]]
    >>> # points: [[0, 0], [1, 0], [2, 2], [3, 2]]
    """
    if not data:
        raise ValueError("Cannot concatenate empty list")

    if isinstance(data[0], Mesh):
        if not all(isinstance(item, Mesh) for item in data):
            raise TypeError(
                "All items must be Mesh objects when first item is Mesh")
        data_list = [(m.faces, _apply_transformation(
            m.points, m.transformation)) for m in data]
    elif isinstance(data[0], EdgeMesh):
        if not all(isinstance(item, EdgeMesh) for item in data):
            raise TypeError(
                "All items must be EdgeMesh objects when first item is EdgeMesh")
        data_list = [(e.edges, _apply_transformation(
            e.points, e.transformation)) for e in data]
    else:
        if not all(isinstance(item, tuple) and len(item) == 2 for item in data):
            raise TypeError(
                "All items must be (indices, points) tuples when first item is tuple")
        data_list = data

    # Validate dims and analyze index types
    first_indices, first_points = data_list[0]
    dims = first_points.shape[1]

    for i, (indices, points) in enumerate(data_list[1:], start=1):
        if points.shape[1] != dims:
            raise ValueError(
                f"All points must have same dims. "
                f"First item has dims={dims}, but item {i} has dims={points.shape[1]}"
            )

    # Determine if we need dynamic output:
    # - Any item is already dynamic (OffsetBlockedArray)
    # - Fixed-size items have different V values (mixed ngon)
    has_dynamic = any(isinstance(idx, OffsetBlockedArray)
                      for idx, _ in data_list)
    fixed_items = [(idx, pts) for idx, pts in data_list if not isinstance(
        idx, OffsetBlockedArray)]

    if fixed_items:
        v_values = set(idx.shape[1] for idx, _ in fixed_items)
        mixed_ngon = len(v_values) > 1
    else:
        mixed_ngon = False

    use_dynamic = has_dynamic or mixed_ngon

    # Calculate cumulative point offsets using numpy for efficiency
    point_counts = np.array([len(points) for _, points in data_list])
    point_offsets = np.concatenate([[0], np.cumsum(point_counts[:-1])])

    # Concatenate points
    concatenated_points = np.concatenate(
        [points for _, points in data_list], axis=0)

    if use_dynamic:
        # Dynamic output: convert all to OffsetBlockedArray format
        all_data = []
        all_block_sizes = []

        for (indices, _), pt_offset in zip(data_list, point_offsets):
            if isinstance(indices, OffsetBlockedArray):
                # Already dynamic
                all_data.append(indices.data + pt_offset)
                all_block_sizes.append(np.diff(indices.offsets))
            else:
                # Fixed-size ndarray - flatten and convert
                all_data.append(indices.ravel() + pt_offset)
                V = indices.shape[1]
                all_block_sizes.append(np.full(len(indices), V, dtype=np.intp))

        # Concatenate data and build offsets
        concatenated_data = np.concatenate(all_data)
        block_sizes = np.concatenate(all_block_sizes)
        concatenated_offsets = np.concatenate([[0], np.cumsum(block_sizes)])

        # Match dtype to input
        out_dtype = first_indices.dtype if isinstance(
            first_indices, OffsetBlockedArray) else first_indices.dtype
        concatenated_offsets = concatenated_offsets.astype(out_dtype)
        concatenated_data = concatenated_data.astype(out_dtype)

        return OffsetBlockedArray(concatenated_offsets, concatenated_data), concatenated_points
    else:
        # Fixed-size case: all same V, simple concatenation
        offset_indices = [
            indices + offset for (indices, _), offset in zip(data_list, point_offsets)]
        concatenated_indices = np.concatenate(offset_indices, axis=0)
        return concatenated_indices, concatenated_points
