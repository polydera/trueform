"""
EdgeMesh data structures for trueform

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from .._core import OffsetBlockedArray
from ._validation import (
    validate_points,
    validate_points_update,
    validate_index_array,
    validate_index_update,
    validate_transformation,
)
from .._trueform.spatial import (
    EdgeMeshWrapperIntFloat2D,
    EdgeMeshWrapperIntFloat3D,
    EdgeMeshWrapperIntDouble2D,
    EdgeMeshWrapperIntDouble3D,
    EdgeMeshWrapperInt64Float2D,
    EdgeMeshWrapperInt64Float3D,
    EdgeMeshWrapperInt64Double2D,
    EdgeMeshWrapperInt64Double3D,
)


# Wrapper lookup: (index_type, real_type, dims) -> wrapper_class
_EDGE_MESH_WRAPPERS = {
    ("Int", "Float", 2): EdgeMeshWrapperIntFloat2D,
    ("Int", "Float", 3): EdgeMeshWrapperIntFloat3D,
    ("Int", "Double", 2): EdgeMeshWrapperIntDouble2D,
    ("Int", "Double", 3): EdgeMeshWrapperIntDouble3D,
    ("Int64", "Float", 2): EdgeMeshWrapperInt64Float2D,
    ("Int64", "Float", 3): EdgeMeshWrapperInt64Float3D,
    ("Int64", "Double", 2): EdgeMeshWrapperInt64Double2D,
    ("Int64", "Double", 3): EdgeMeshWrapperInt64Double3D,
}


class EdgeMesh:
    """
    Edge mesh with spatial indexing support.

    Wraps NumPy arrays of edges and points and provides efficient spatial queries
    through an internal tree structure. Edges have topology - they share vertices
    and can be queried for connectivity.

    Parameters
    ----------
    edges : np.ndarray
        Array of shape (N, 2) where N is number of edges. Each edge is a pair of
        vertex indices. Supports int32 and int64 dtypes.
    points : np.ndarray
        Array of shape (P, D) where P is number of points and D is dimensionality (2 or 3).
        Supports float32 and float64 dtypes.

    Examples
    --------
    >>> import numpy as np
    >>> import trueform as tf
    >>> # Edge mesh in 3D with float32 (connected line segments)
    >>> edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    >>> points = np.random.rand(4, 3).astype(np.float32)
    >>> edge_mesh = tf.EdgeMesh(edges, points)
    >>> edge_mesh.number_of_points
    4
    >>> edge_mesh.number_of_edges
    3
    >>> edge_mesh.dims
    3
    """

    def __init__(
        self, edges: np.ndarray, points: np.ndarray, transformation: np.ndarray = None
    ):
        """
        Create an edge mesh from edge and point NumPy arrays.

        Parameters
        ----------
        edges : np.ndarray
            Array of shape (N, 2) with dtype int32 or int64
        points : np.ndarray
            Array of shape (P, D) where D is 2 or 3, with dtype float32 or float64
        transformation : np.ndarray, optional
            Transformation matrix (3x3 for 2D, 4x4 for 3D). If provided, applies
            transformation to points during spatial queries.
        """
        # Validate and normalize inputs
        edges = validate_index_array(edges, expected_cols=2, name="edges")
        points, dims = validate_points(points)

        # Store arrays (Python owns this data)
        self._edges = edges
        self._points = points

        # Deduce wrapper type from dtypes and dims
        index_type = "Int" if edges.dtype == np.int32 else "Int64"
        real_type = "Float" if points.dtype == np.float32 else "Double"
        key = (index_type, real_type, dims)

        # Look up the wrapper class
        wrapper_class = _EDGE_MESH_WRAPPERS.get(key)
        if wrapper_class is None:
            raise ValueError(
                f"Unsupported combination: edges dtype={edges.dtype}, "
                f"points dtype={points.dtype}, dims={dims}"
            )

        # Create wrapper
        self._wrapper = wrapper_class(edges, points)

        # Set transformation if provided
        if transformation is not None:
            self.transformation = transformation

    @property
    def edges(self) -> np.ndarray:
        """Get the underlying edges array."""
        return self._edges

    @edges.setter
    def edges(self, value: np.ndarray) -> None:
        """
        Set the underlying edges array.

        Automatically marks the edge mesh as modified.

        Parameters
        ----------
        value : np.ndarray
            New edges array. Must have same dtype and shape[1]==2.
        """
        value = validate_index_update(value, self._edges.dtype, expected_cols=2, name="Edges")
        self._edges = value
        self._wrapper.set_edges_array(value)

    @property
    def points(self) -> np.ndarray:
        """Get the underlying points array."""
        return self._points

    @points.setter
    def points(self, value: np.ndarray) -> None:
        """
        Set the underlying points array.

        Automatically marks the edge mesh as modified.

        Parameters
        ----------
        value : np.ndarray
            New points array. Must have same dtype and dimensionality as original.
        """
        value = validate_points_update(value, self._points.dtype, self._points.shape[1])
        self._points = value
        self._wrapper.set_points_array(value)

    @property
    def number_of_points(self) -> int:
        """Get number of points in the edge mesh."""
        return len(self._points)

    @property
    def number_of_edges(self) -> int:
        """Get number of edges in the edge mesh."""
        return len(self._edges)

    @property
    def dims(self) -> int:
        """Get dimensionality of points."""
        return self._wrapper.dims()

    @property
    def dtype(self) -> np.dtype:
        """Get data type of points (float32 or float64)."""
        return self._points.dtype

    @property
    def transformation(self):
        """
        Get the transformation matrix.

        Returns
        -------
        np.ndarray or None
            Transformation matrix (3x3 for 2D, 4x4 for 3D), or None if not set
        """
        return self._wrapper.transformation()

    @transformation.setter
    def transformation(self, mat: np.ndarray) -> None:
        """
        Set the transformation matrix.

        Parameters
        ----------
        mat : np.ndarray or None
            Transformation matrix (3x3 for 2D points, 4x4 for 3D points).
            Set to None to clear the transformation.
        """
        if mat is None:
            self._wrapper.clear_transformation()
            return

        mat = validate_transformation(mat, self.dims, self._points.dtype)
        self._wrapper.set_transformation(mat)

    def build_tree(self) -> None:
        """
        Build the spatial index tree.

        Call this after modifying the points or edges arrays to update the spatial index.
        """
        self._wrapper.build_tree()

    def build_edge_membership(self) -> None:
        """
        Build the edge membership structure.

        Call this after modifying the edges array to update the edge membership.
        """
        self._wrapper.build_edge_membership()

    @property
    def edge_membership(self):
        """
        Get the edge membership structure.

        For each vertex, contains all edges containing that vertex.

        Builds the structure if not already built.

        Returns
        -------
        OffsetBlockedArray
            Edge membership mapping vertices to edges containing them.
        """
        wrapper = self._wrapper.edge_membership_array()
        return OffsetBlockedArray(wrapper.offsets_array(), wrapper.data_array())

    @edge_membership.setter
    def edge_membership(self, value: "OffsetBlockedArray") -> None:
        """
        Set the edge membership structure.

        Parameters
        ----------
        value : OffsetBlockedArray
            Edge membership structure.
        """
        self._wrapper.set_edge_membership(value._wrapper)

    def build_vertex_link(self) -> None:
        """
        Build the vertex link structure.

        Call this after modifying the edges array to update the vertex link.
        """
        self._wrapper.build_vertex_link()

    @property
    def vertex_link(self):
        """
        Get the vertex link structure.

        For each vertex, contains all other vertices that share an edge with it.

        Builds the structure if not already built.

        Returns
        -------
        OffsetBlockedArray
            Vertex link mapping vertices to connected vertices.
        """
        wrapper = self._wrapper.vertex_link_array()
        return OffsetBlockedArray(wrapper.offsets_array(), wrapper.data_array())

    @vertex_link.setter
    def vertex_link(self, value: "OffsetBlockedArray") -> None:
        """
        Set the vertex link structure.

        Parameters
        ----------
        value : OffsetBlockedArray
            Vertex link structure.
        """
        self._wrapper.set_vertex_link(value._wrapper)

    def shared_view(self) -> "EdgeMesh":
        """
        Create a new EdgeMesh instance sharing the same underlying data.

        The new edge mesh shares the same edges, points, and cached structures (tree,
        edge_membership, vertex_link) but has its own transformation.
        This is useful when you need multiple edge mesh instances with different
        transformations but the same geometry.

        Returns
        -------
        EdgeMesh
            New edge mesh instance sharing the same data, without transformation.

        Examples
        --------
        >>> edge_mesh = tf.EdgeMesh(edges, points)
        >>> edge_mesh.transformation = transform_A
        >>> edge_mesh2 = edge_mesh.shared_view()
        >>> edge_mesh2.transformation = transform_B
        >>> # edge_mesh and edge_mesh2 share the same data but have different transforms
        """
        new_edge_mesh = object.__new__(EdgeMesh)
        new_edge_mesh._edges = self._edges
        new_edge_mesh._points = self._points
        new_edge_mesh._wrapper = self._wrapper.shared_view()
        return new_edge_mesh

    def __repr__(self) -> str:
        """String representation of the edge mesh."""
        return f"EdgeMesh({self.number_of_points} points, {self.number_of_edges} edges, {self.dims}D, dtype={self.dtype})"
