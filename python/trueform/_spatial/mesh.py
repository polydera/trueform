"""
Mesh data structures for trueform

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

import numpy as np
from typing import Union
from .._core import OffsetBlockedArray
from .._trueform.spatial import (
    MeshWrapperIntFloat32D,
    MeshWrapperIntFloat33D,
    MeshWrapperIntFloat42D,
    MeshWrapperIntFloat43D,
    MeshWrapperIntDouble32D,
    MeshWrapperIntDouble33D,
    MeshWrapperIntDouble42D,
    MeshWrapperIntDouble43D,
    MeshWrapperInt64Float32D,
    MeshWrapperInt64Float33D,
    MeshWrapperInt64Float42D,
    MeshWrapperInt64Float43D,
    MeshWrapperInt64Double32D,
    MeshWrapperInt64Double33D,
    MeshWrapperInt64Double42D,
    MeshWrapperInt64Double43D,
)


class Mesh:
    """
    Mesh with spatial indexing support.

    Wraps NumPy arrays of faces and points and provides efficient spatial queries
    through an internal tree structure.

    Parameters
    ----------
    faces : np.ndarray
        Array of shape (N, M) where N is number of faces and M is 3 (triangles) or 4 (quads).
        Supports int32 and int64 dtypes.
    points : np.ndarray
        Array of shape (P, D) where P is number of points and D is dimensionality (2 or 3).
        Supports float32 and float64 dtypes.

    Examples
    --------
    >>> import numpy as np
    >>> import trueform as tf
    >>> # Triangle mesh in 3D with float32
    >>> faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    >>> points = np.random.rand(4, 3).astype(np.float32)
    >>> mesh = tf.Mesh(faces, points)
    >>> mesh.number_of_points
    4
    >>> mesh.dims
    3
    """

    def __init__(
        self, faces: np.ndarray, points: np.ndarray, transformation: np.ndarray = None
    ):
        """
        Create a mesh from face and point NumPy arrays.

        Parameters
        ----------
        faces : np.ndarray
            Array of shape (N, M) where M is 3 or 4, with dtype int32 or int64
        points : np.ndarray
            Array of shape (P, D) where D is 2 or 3, with dtype float32 or float64
        transformation : np.ndarray, optional
            Transformation matrix (3x3 for 2D, 4x4 for 3D). If provided, applies
            transformation to points during spatial queries.
        """
        # Validate faces
        if not isinstance(faces, np.ndarray):
            raise TypeError(
                f"Expected numpy array for faces, got {type(faces)}")
        if faces.ndim != 2:
            raise ValueError(
                f"Expected 2D array for faces, got shape {faces.shape}")

        # Validate points
        if not isinstance(points, np.ndarray):
            raise TypeError(
                f"Expected numpy array for points, got {type(points)}")
        if points.ndim != 2:
            raise ValueError(
                f"Expected 2D array for points, got shape {points.shape}")

        # Check face dtype
        if faces.dtype not in [np.int32, np.int64]:
            raise TypeError(
                f"Face indices must be int32 or int64, got {faces.dtype}. "
                f"Convert with faces.astype(np.int32) or faces.astype(np.int64)"
            )

        # Check point dtype
        if points.dtype not in [np.float32, np.float64]:
            # Try to convert to float32
            points = points.astype(np.float32)

        # Check Ngon (triangles or quads)
        ngon = faces.shape[1]
        if ngon not in [3, 4]:
            raise ValueError(
                f"Faces must have 3 (triangles) or 4 (quads) vertices, got {ngon}"
            )

        # Check dimensionality
        dims = points.shape[1]
        if dims not in [2, 3]:
            raise ValueError(
                f"Points must be 2D or 3D, got {dims} dimensions"
            )

        # Ensure C-contiguous layout for zero-copy views
        if not faces.flags["C_CONTIGUOUS"]:
            faces = np.ascontiguousarray(faces)
        if not points.flags["C_CONTIGUOUS"]:
            points = np.ascontiguousarray(points)

        # Store arrays (Python owns this data)
        self._faces = faces
        self._points = points

        # Deduce wrapper type from dtypes and shapes
        index_type = "Int" if faces.dtype == np.int32 else "Int64"
        real_type = "Float" if points.dtype == np.float32 else "Double"
        wrapper_name = f"MeshWrapper{index_type}{real_type}{ngon}{dims}D"

        # Look up the wrapper class
        wrapper_class = globals().get(wrapper_name)
        if wrapper_class is None:
            raise ValueError(
                f"Unsupported combination: faces dtype={faces.dtype}, "
                f"points dtype={points.dtype}, ngon={ngon}, dims={dims}"
            )

        # Create wrapper
        self._wrapper = wrapper_class(faces, points)

        # Set transformation if provided
        if transformation is not None:
            self.transformation = transformation

    @property
    def faces(self) -> np.ndarray:
        """Get the underlying faces array."""
        return self._faces

    @property
    def points(self) -> np.ndarray:
        """Get the underlying points array."""
        return self._points

    @property
    def number_of_points(self) -> int:
        """Get number of points in the mesh."""
        return len(self._points)

    @property
    def number_of_faces(self) -> int:
        """Get number of faces in the mesh."""
        return len(self._faces)

    @property
    def dims(self) -> int:
        """Get dimensionality of points."""
        return self._wrapper.dims()

    @property
    def ngon(self) -> int:
        """Get number of vertices per face (3 for triangles, 4 for quads)."""
        return self._faces.shape[1]

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

        Call this after modifying the points or faces arrays to update the spatial index.
        """
        self._wrapper.rebuild_tree()

    def build_face_membership(self) -> None:
        """
        Build the face membership structure.

        Call this after modifying the faces array to update the face membership.
        """
        self._wrapper.rebuild_face_membership()

    def build_manifold_edge_link(self) -> None:
        """
        Build the manifold edge link structure.

        Call this after modifying the faces array to update the manifold edge link.
        Requires face membership to be built first (calls build_face_membership if needed).
        """
        self._wrapper.rebuild_manifold_edge_link()

    @property
    def face_membership(self):
        """
        Get the face membership structure.

        Builds the structure if not already built.

        Returns
        -------
        OffsetBlockedArray
            Face membership mapping points to faces.
        """
        wrapper = self._wrapper.face_membership_array()
        return OffsetBlockedArray(wrapper.offsets_array(), wrapper.data_array())

    @face_membership.setter
    def face_membership(self, value) -> None:
        """
        Set the face membership structure.

        Parameters
        ----------
        value : OffsetBlockedArray or None
            Face membership structure. Set to None to clear.
        """
        if value is None:
            self._wrapper.clear_face_membership()
            return

        # Pass the wrapper to C++
        self._wrapper.set_face_membership(value._wrapper)

    @property
    def manifold_edge_link(self):
        """
        Get the manifold edge link array.

        Builds the structure if not already built (also builds face_membership if needed).

        Returns
        -------
        np.ndarray
            Manifold edge link array, shape (num_faces, ngon) with dtype matching faces.
        """
        return self._wrapper.manifold_edge_link_array()

    @manifold_edge_link.setter
    def manifold_edge_link(self, arr: np.ndarray) -> None:
        """
        Set the manifold edge link array.

        Parameters
        ----------
        arr : np.ndarray or None
            Manifold edge link array, shape (num_faces, ngon). Set to None to clear.
        """
        if arr is None:
            self._wrapper.clear_manifold_edge_link()
            return

        # Ensure C-contiguous
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        self._wrapper.set_manifold_edge_link(arr)

    def build_face_link(self) -> None:
        """
        Build the face link structure.

        Call this after modifying the faces array to update the face link.
        Requires face membership to be built first (calls build_face_membership if needed).
        """
        self._wrapper.rebuild_face_link()

    def build_vertex_link(self) -> None:
        """
        Build the vertex link structure.

        Call this after modifying the faces array to update the vertex link.
        Requires face membership to be built first (calls build_face_membership if needed).
        """
        self._wrapper.rebuild_vertex_link()

    @property
    def face_link(self):
        """
        Get the face link structure.

        For each vertex, contains all faces in the vertex's link (faces sharing
        an edge with any face containing that vertex).

        Builds the structure if not already built (also builds face_membership if needed).

        Returns
        -------
        OffsetBlockedArray
            Face link mapping vertices to faces in their link.
        """
        wrapper = self._wrapper.face_link_array()
        return OffsetBlockedArray(wrapper.offsets_array(), wrapper.data_array())

    @face_link.setter
    def face_link(self, value) -> None:
        """
        Set the face link structure.

        Parameters
        ----------
        value : OffsetBlockedArray or None
            Face link structure. Set to None to clear.
        """
        if value is None:
            self._wrapper.clear_face_link()
            return

        # Pass the wrapper to C++
        self._wrapper.set_face_link(value._wrapper)

    @property
    def vertex_link(self):
        """
        Get the vertex link structure.

        For each vertex, contains all other vertices that share a face with it.

        Builds the structure if not already built (also builds face_membership if needed).

        Returns
        -------
        OffsetBlockedArray
            Vertex link mapping vertices to connected vertices.
        """
        wrapper = self._wrapper.vertex_link_array()
        return OffsetBlockedArray(wrapper.offsets_array(), wrapper.data_array())

    @vertex_link.setter
    def vertex_link(self, value) -> None:
        """
        Set the vertex link structure.

        Parameters
        ----------
        value : OffsetBlockedArray or None
            Vertex link structure. Set to None to clear.
        """
        if value is None:
            self._wrapper.clear_vertex_link()
            return

        # Pass the wrapper to C++
        self._wrapper.set_vertex_link(value._wrapper)

    def __repr__(self) -> str:
        """String representation of the mesh."""
        return f"Mesh({self.number_of_points} points, {self.number_of_faces} faces, {self.ngon}-gon, {self.dims}D, dtype={self.dtype})"
