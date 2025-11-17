"""
Concatenate multiple meshes or edge meshes into a single geometry.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

from typing import List, Tuple, Union
import numpy as np
from .._core import Mesh, EdgeMesh


def concatenated(
    data: Union[List[Tuple[np.ndarray, np.ndarray]], List[Mesh], List[EdgeMesh]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatenate multiple meshes or edge meshes into a single geometry.

    Takes a list of geometric data and merges them into a single unified structure,
    automatically handling index offsetting to maintain referential integrity.

    Parameters
    ----------
    data : List[Tuple[np.ndarray, np.ndarray]] | List[Mesh] | List[EdgeMesh]
        List of geometries to concatenate. Can be:
        - List of (indices, points) tuples
        - List of Mesh objects
        - List of EdgeMesh objects

        All geometries must have:
        - Same V (vertices per primitive): all triangles, all quads, or all edges
        - Same dims (point dimensions): all 2D or all 3D

        Dtypes can be mixed - numpy will handle type promotion automatically.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (concatenated_indices, concatenated_points)
        - concatenated_indices: Combined indices with offsets applied
        - concatenated_points: Combined point coordinates

    Raises
    ------
    ValueError
        If input list is empty, or if V or dims don't match across all inputs
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

    # Normalize input - extract indices and points from Mesh/EdgeMesh objects
    if isinstance(data[0], Mesh):
        # Check all are Mesh
        if not all(isinstance(item, Mesh) for item in data):
            raise TypeError("All items must be Mesh objects when first item is Mesh")
        data_list = [(mesh.faces, mesh.points) for mesh in data]
    elif isinstance(data[0], EdgeMesh):
        # Check all are EdgeMesh
        if not all(isinstance(item, EdgeMesh) for item in data):
            raise TypeError("All items must be EdgeMesh objects when first item is EdgeMesh")
        data_list = [(edgemesh.edges, edgemesh.points) for edgemesh in data]
    else:
        # Assume list of tuples
        if not all(isinstance(item, tuple) and len(item) == 2 for item in data):
            raise TypeError("All items must be (indices, points) tuples when first item is tuple")
        data_list = data

    # Validate all have same V (vertices per primitive)
    first_indices, first_points = data_list[0]
    V = first_indices.shape[1]
    dims = first_points.shape[1]

    for i, (indices, points) in enumerate(data_list[1:], start=1):
        if indices.shape[1] != V:
            raise ValueError(
                f"All indices must have same V (vertices per primitive). "
                f"First item has V={V}, but item {i} has V={indices.shape[1]}"
            )
        if points.shape[1] != dims:
            raise ValueError(
                f"All points must have same dims. "
                f"First item has dims={dims}, but item {i} has dims={points.shape[1]}"
            )

    # Calculate cumulative offsets for point indices
    offsets = [0]
    for indices, points in data_list[:-1]:
        offsets.append(offsets[-1] + len(points))

    # Offset and concatenate indices
    offset_indices = [indices + offset for (indices, points), offset in zip(data_list, offsets)]
    concatenated_indices = np.concatenate(offset_indices, axis=0)

    # Concatenate points
    concatenated_points = np.concatenate([points for _, points in data_list], axis=0)

    return concatenated_indices, concatenated_points
