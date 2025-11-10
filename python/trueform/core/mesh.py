"""
Mesh data structures for trueform

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

import numpy as np
from typing import Union
from .._trueform import (
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
    >>> mesh.size()
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
            raise TypeError(f"Expected numpy array for faces, got {type(faces)}")
        if faces.ndim != 2:
            raise ValueError(f"Expected 2D array for faces, got shape {faces.shape}")

        # Validate points
        if not isinstance(points, np.ndarray):
            raise TypeError(f"Expected numpy array for points, got {type(points)}")
        if points.ndim != 2:
            raise ValueError(f"Expected 2D array for points, got shape {points.shape}")

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
        mat : np.ndarray
            Transformation matrix (3x3 for 2D points, 4x4 for 3D points)
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

    def has_transformation(self) -> bool:
        """Check if a transformation matrix is set."""
        return self._wrapper.has_transformation()

    def clear_transformation(self) -> None:
        """Clear the transformation matrix."""
        self._wrapper.clear_transformation()

    def size(self) -> int:
        """Get number of points in the mesh."""
        return self._wrapper.size()

    @property
    def dims(self) -> int:
        """Get dimensionality of points."""
        return self._wrapper.dims()

    @property
    def ngon(self) -> int:
        """Get number of vertices per face (3 for triangles, 4 for quads)."""
        return self._faces.shape[1]

    def rebuild_tree(self) -> None:
        """
        Rebuild the spatial index tree.

        Call this after modifying the points or faces arrays to update the spatial index.
        This will rebuild the tree even if one already exists.
        """
        self._wrapper.rebuild_tree()

    def ensure_tree(self) -> None:
        """
        Build the spatial index tree if not already built.

        The tree is built lazily - it's only created when needed for
        spatial queries. This method can be called explicitly to build
        the tree ahead of time.
        """
        self._wrapper.ensure_tree()

    def clear_tree(self) -> None:
        """
        Clear the spatial index tree to free memory.

        The tree will be automatically rebuilt if needed for future queries.
        """
        self._wrapper.clear_tree()

    def has_tree(self) -> bool:
        """Check if the spatial index tree is currently built."""
        return self._wrapper.has_tree()

    def rebuild_face_membership(self) -> None:
        """
        Rebuild the face membership structure.

        Call this after modifying the faces array to update the face membership.
        This will rebuild the structure even if one already exists.
        """
        self._wrapper.rebuild_face_membership()

    def ensure_face_membership(self) -> None:
        """
        Build the face membership structure if not already built.

        The face membership is built lazily - it's only created when needed.
        This method can be called explicitly to build it ahead of time.
        """
        self._wrapper.ensure_face_membership()

    def clear_face_membership(self) -> None:
        """
        Clear the face membership structure to free memory.

        The structure will be automatically rebuilt if needed for future operations.
        """
        self._wrapper.clear_face_membership()

    def has_face_membership(self) -> bool:
        """Check if the face membership structure is currently built."""
        return self._wrapper.has_face_membership()

    def __len__(self) -> int:
        """Get number of points (for len(mesh) syntax)."""
        return self.size()

    def __repr__(self) -> str:
        """String representation of the mesh."""
        tree_status = "with tree" if self.has_tree() else "no tree"
        fm_status = "with face_membership" if self.has_face_membership() else "no face_membership"
        return f"Mesh({self.size()} points, {len(self._faces)} faces, {self.ngon}-gon, {self.dims}D, {tree_status}, {fm_status})"
