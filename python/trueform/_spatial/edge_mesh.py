"""
EdgeMesh data structures for trueform

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

import numpy as np
from typing import Union
from .._trueform import (
    EdgeMeshWrapperIntFloat2D,
    EdgeMeshWrapperIntFloat3D,
    EdgeMeshWrapperIntDouble2D,
    EdgeMeshWrapperIntDouble3D,
    EdgeMeshWrapperInt64Float2D,
    EdgeMeshWrapperInt64Float3D,
    EdgeMeshWrapperInt64Double2D,
    EdgeMeshWrapperInt64Double3D,
)


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
        # Validate edges
        if not isinstance(edges, np.ndarray):
            raise TypeError(
                f"Expected numpy array for edges, got {type(edges)}")
        if edges.ndim != 2:
            raise ValueError(
                f"Expected 2D array for edges, got shape {edges.shape}")
        if edges.shape[1] != 2:
            raise ValueError(
                f"Edges must have 2 vertices per edge, got {edges.shape[1]}"
            )

        # Validate points
        if not isinstance(points, np.ndarray):
            raise TypeError(
                f"Expected numpy array for points, got {type(points)}")
        if points.ndim != 2:
            raise ValueError(
                f"Expected 2D array for points, got shape {points.shape}")

        # Check edge dtype
        if edges.dtype not in [np.int32, np.int64]:
            raise TypeError(
                f"Edge indices must be int32 or int64, got {edges.dtype}. "
                f"Convert with edges.astype(np.int32) or edges.astype(np.int64)"
            )

        # Check point dtype
        if points.dtype not in [np.float32, np.float64]:
            # Try to convert to float32
            points = points.astype(np.float32)

        # Check dimensionality
        dims = points.shape[1]
        if dims not in [2, 3]:
            raise ValueError(
                f"Points must be 2D or 3D, got {dims} dimensions"
            )

        # Ensure C-contiguous layout for zero-copy views
        if not edges.flags["C_CONTIGUOUS"]:
            edges = np.ascontiguousarray(edges)
        if not points.flags["C_CONTIGUOUS"]:
            points = np.ascontiguousarray(points)

        # Store arrays (Python owns this data)
        self._edges = edges
        self._points = points

        # Deduce wrapper type from dtypes and dims
        index_type = "Int" if edges.dtype == np.int32 else "Int64"
        real_type = "Float" if points.dtype == np.float32 else "Double"
        wrapper_name = f"EdgeMeshWrapper{index_type}{real_type}{dims}D"

        # Look up the wrapper class
        wrapper_class = globals().get(wrapper_name)
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

    @property
    def points(self) -> np.ndarray:
        """Get the underlying points array."""
        return self._points

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

        # Validate matrix shape
        expected_size = self.dims + 1
        if mat.shape != (expected_size, expected_size):
            raise ValueError(
                f"Transformation must be {expected_size}x{expected_size} for {self.dims}D points, "
                f"got shape {mat.shape}"
            )

        # Validate dtype matches points
        if mat.dtype != self._points.dtype:
            raise TypeError(
                f"Transformation dtype ({mat.dtype}) must match points dtype ({self._points.dtype})"
            )

        # Ensure C-contiguous
        if not mat.flags["C_CONTIGUOUS"]:
            mat = np.ascontiguousarray(mat)

        self._wrapper.set_transformation(mat)

    def build_tree(self) -> None:
        """
        Build the spatial index tree.

        Call this after modifying the points or edges arrays to update the spatial index.
        """
        self._wrapper.rebuild_tree()

    def build_edge_membership(self) -> None:
        """
        Build the edge membership structure.

        Call this after modifying the edges array to update the edge membership.
        """
        self._wrapper.rebuild_edge_membership()

    def __repr__(self) -> str:
        """String representation of the edge mesh."""
        return f"EdgeMesh({self.number_of_points} points, {self.number_of_edges} edges, {self.dims}D, dtype={self.dtype})"
