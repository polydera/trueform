"""
Mesh data structures for trueform

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Optional
from .._core import OffsetBlockedArray
from ._validation import (
    validate_points,
    validate_points_update,
    validate_transformation,
    ensure_contiguous,
)
from .._trueform.spatial import (
    # Fixed-size meshes (triangles): int32
    MeshWrapperIntFloat32D,
    MeshWrapperIntFloat33D,
    MeshWrapperIntDouble32D,
    MeshWrapperIntDouble33D,
    # Fixed-size meshes (triangles): int64
    MeshWrapperInt64Float32D,
    MeshWrapperInt64Float33D,
    MeshWrapperInt64Double32D,
    MeshWrapperInt64Double33D,
    # Dynamic-size meshes: int32
    MeshWrapperIntFloatDynamic2D,
    MeshWrapperIntFloatDynamic3D,
    MeshWrapperIntDoubleDynamic2D,
    MeshWrapperIntDoubleDynamic3D,
    # Dynamic-size meshes: int64
    MeshWrapperInt64FloatDynamic2D,
    MeshWrapperInt64FloatDynamic3D,
    MeshWrapperInt64DoubleDynamic2D,
    MeshWrapperInt64DoubleDynamic3D,
)


# Lookup tables for wrapper classes
# Only triangles (ngon=3) supported for fixed-size meshes
# Use OffsetBlockedArray for variable-sized polygons (dynamic)
_FIXED_SIZE_WRAPPERS = {
    ("Int", "Float", 3, 2): MeshWrapperIntFloat32D,
    ("Int", "Float", 3, 3): MeshWrapperIntFloat33D,
    ("Int", "Double", 3, 2): MeshWrapperIntDouble32D,
    ("Int", "Double", 3, 3): MeshWrapperIntDouble33D,
    ("Int64", "Float", 3, 2): MeshWrapperInt64Float32D,
    ("Int64", "Float", 3, 3): MeshWrapperInt64Float33D,
    ("Int64", "Double", 3, 2): MeshWrapperInt64Double32D,
    ("Int64", "Double", 3, 3): MeshWrapperInt64Double33D,
}

_DYNAMIC_SIZE_WRAPPERS = {
    ("Int", "Float", 2): MeshWrapperIntFloatDynamic2D,
    ("Int", "Float", 3): MeshWrapperIntFloatDynamic3D,
    ("Int", "Double", 2): MeshWrapperIntDoubleDynamic2D,
    ("Int", "Double", 3): MeshWrapperIntDoubleDynamic3D,
    ("Int64", "Float", 2): MeshWrapperInt64FloatDynamic2D,
    ("Int64", "Float", 3): MeshWrapperInt64FloatDynamic3D,
    ("Int64", "Double", 2): MeshWrapperInt64DoubleDynamic2D,
    ("Int64", "Double", 3): MeshWrapperInt64DoubleDynamic3D,
}


class Mesh:
    """
    Mesh with spatial indexing support.

    Wraps NumPy arrays of faces and points and provides efficient spatial queries
    through an internal tree structure.

    Parameters
    ----------
    faces : np.ndarray or OffsetBlockedArray
        For triangle meshes: Array of shape (N, 3) where N is number of faces.
        Supports int32 and int64 dtypes.
        For dynamic-size faces (n-gons): OffsetBlockedArray containing variable-
        sized polygons.
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

    >>> # Dynamic-size mesh (n-gons)
    >>> offsets = np.array([0, 3, 7, 10], dtype=np.int32)  # 3, 4, 3 vertices
    >>> data = np.array([0, 1, 2, 0, 1, 2, 3, 0, 2, 3], dtype=np.int32)
    >>> faces = tf.OffsetBlockedArray(offsets, data)
    >>> mesh = tf.Mesh(faces, points)
    """

    def __init__(
        self,
        faces: "np.ndarray | OffsetBlockedArray",
        points: np.ndarray,
        transformation: np.ndarray = None
    ):
        """
        Create a mesh from face and point NumPy arrays.

        Parameters
        ----------
        faces : np.ndarray or OffsetBlockedArray
            For triangles: Array of shape (N, 3) with dtype int32 or int64
            For dynamic-size: OffsetBlockedArray with variable-sized polygons
        points : np.ndarray
            Array of shape (P, D) where D is 2 or 3, with dtype float32 or float64
        transformation : np.ndarray, optional
            Transformation matrix (3x3 for 2D, 4x4 for 3D). If provided, applies
            transformation to points during spatial queries.
        """
        # Validate and normalize points
        points, dims = validate_points(points)
        self._points = points

        # Determine index and real types
        real_type = "Float" if points.dtype == np.float32 else "Double"

        # Check if dynamic-size (OffsetBlockedArray) or fixed-size (ndarray)
        if isinstance(faces, OffsetBlockedArray):
            # Dynamic-size mesh
            self._is_dynamic = True
            self._faces = faces

            # Determine index type from OffsetBlockedArray
            index_dtype = faces.data.dtype
            if index_dtype not in [np.int32, np.int64]:
                raise TypeError(
                    f"Face indices must be int32 or int64, got {index_dtype}"
                )
            index_type = "Int" if index_dtype == np.int32 else "Int64"

            # Look up dynamic wrapper
            key = (index_type, real_type, dims)
            wrapper_class = _DYNAMIC_SIZE_WRAPPERS.get(key)
            if wrapper_class is None:
                raise ValueError(
                    f"Unsupported combination: faces dtype={index_dtype}, "
                    f"points dtype={points.dtype}, dims={dims}"
                )

            # Create wrapper with OffsetBlockedArray's internal wrapper
            self._wrapper = wrapper_class(faces._wrapper, points)

        elif isinstance(faces, np.ndarray):
            # Fixed-size mesh
            self._is_dynamic = False

            if faces.ndim != 2:
                raise ValueError(
                    f"Expected 2D array for faces, got shape {faces.shape}")

            # Check face dtype
            if faces.dtype not in [np.int32, np.int64]:
                raise TypeError(
                    f"Face indices must be int32 or int64, got {faces.dtype}. "
                    f"Convert with faces.astype(np.int32) or faces.astype(np.int64)"
                )

            # Check Ngon (only triangles supported for fixed-size)
            ngon = faces.shape[1]
            if ngon != 3:
                raise ValueError(
                    f"Fixed-size faces must have 3 vertices (triangles), got {ngon}. "
                    f"For variable-sized polygons, use OffsetBlockedArray."
                )

            # Ensure C-contiguous and store
            faces = ensure_contiguous(faces)
            self._faces = faces

            # Determine index type
            index_type = "Int" if faces.dtype == np.int32 else "Int64"

            # Look up fixed-size wrapper
            key = (index_type, real_type, ngon, dims)
            wrapper_class = _FIXED_SIZE_WRAPPERS.get(key)
            if wrapper_class is None:
                raise ValueError(
                    f"Unsupported combination: faces dtype={faces.dtype}, "
                    f"points dtype={points.dtype}, ngon={ngon}, dims={dims}"
                )

            # Create wrapper
            self._wrapper = wrapper_class(faces, points)

        else:
            raise TypeError(
                f"Expected numpy array or OffsetBlockedArray for faces, got {type(faces)}"
            )

        # Set transformation if provided
        if transformation is not None:
            self.transformation = transformation

    @property
    def faces(self) -> "np.ndarray | OffsetBlockedArray":
        """Get the underlying faces array or OffsetBlockedArray."""
        return self._faces

    @faces.setter
    def faces(self, value: "np.ndarray | OffsetBlockedArray") -> None:
        """
        Set the underlying faces.

        Automatically marks the mesh as modified.

        Parameters
        ----------
        value : np.ndarray or OffsetBlockedArray
            New faces. Must match original type (fixed vs dynamic).
        """
        if self._is_dynamic:
            if not isinstance(value, OffsetBlockedArray):
                raise TypeError(
                    f"Faces must be OffsetBlockedArray for dynamic mesh, got {type(value)}"
                )
            self._faces = value
            self._wrapper.set_faces_array(value._wrapper)
        else:
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"Faces must be numpy array for fixed-size mesh, got {type(value)}"
                )
            if value.dtype != self._faces.dtype:
                raise TypeError(
                    f"Faces dtype ({value.dtype}) must match original dtype ({self._faces.dtype})"
                )
            if value.shape[1] != self._faces.shape[1]:
                raise ValueError(
                    f"Faces ngon ({value.shape[1]}) must match original ({self._faces.shape[1]})"
                )
            self._faces = ensure_contiguous(value)
            self._wrapper.set_faces_array(self._faces)

    @property
    def points(self) -> np.ndarray:
        """Get the underlying points array."""
        return self._points

    @points.setter
    def points(self, value: np.ndarray) -> None:
        """
        Set the underlying points array.

        Automatically marks the mesh as modified.

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
        """Get number of points in the mesh."""
        return len(self._points)

    @property
    def number_of_faces(self) -> int:
        """Get number of faces in the mesh."""
        return self._wrapper.number_of_faces()

    @property
    def dims(self) -> int:
        """Get dimensionality of points."""
        return self._wrapper.dims()

    @property
    def ngon(self) -> Optional[int]:
        """
        Get number of vertices per face.

        Returns 3 for triangle meshes, None for dynamic-size meshes.
        """
        if self._is_dynamic:
            return None
        return self._faces.shape[1]

    @property
    def is_dynamic(self) -> bool:
        """Return True if mesh has variable-sized faces."""
        return self._is_dynamic

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

        Builds the tree if not already built or if data has been modified.
        """
        self._wrapper.build_tree()

    def build_face_membership(self) -> None:
        """
        Build the face membership structure.

        Builds the structure if not already built or if data has been modified.
        """
        self._wrapper.build_face_membership()

    def build_manifold_edge_link(self) -> None:
        """
        Build the manifold edge link structure.

        Builds the structure if not already built or if data has been modified.
        Also builds face_membership if needed.
        """
        self._wrapper.build_manifold_edge_link()

    def build_face_link(self) -> None:
        """
        Build the face link structure.

        Builds the structure if not already built or if data has been modified.
        Also builds face_membership if needed.
        """
        self._wrapper.build_face_link()

    def build_vertex_link(self) -> None:
        """
        Build the vertex link structure.

        Builds the structure if not already built or if data has been modified.
        Also builds face_membership if needed.
        """
        self._wrapper.build_vertex_link()

    @property
    def face_membership(self):
        """
        Get the face membership structure.

        For each vertex, contains all faces containing that vertex.

        Builds the structure if not already built.

        Returns
        -------
        OffsetBlockedArray
            Face membership mapping vertices to faces containing them.
        """
        wrapper = self._wrapper.face_membership_array()
        return OffsetBlockedArray(wrapper.offsets_array(), wrapper.data_array())

    @face_membership.setter
    def face_membership(self, value: OffsetBlockedArray) -> None:
        """
        Set the face membership structure.

        Parameters
        ----------
        value : OffsetBlockedArray
            Face membership structure.
        """
        self._wrapper.set_face_membership(value._wrapper)

    @property
    def manifold_edge_link(self) -> "np.ndarray | OffsetBlockedArray":
        """
        Get the manifold edge link array.

        For each face and edge, contains the index of the adjacent face sharing that edge.
        Special values:
        - >= 0: index of adjacent face
        - -1: boundary edge (no adjacent face)
        - -2: non-manifold edge (shared by more than 2 faces)
        - -3: non-manifold representative

        Builds the structure if not already built (also builds face_membership if needed).

        Returns
        -------
        np.ndarray or OffsetBlockedArray
            For fixed-size: array of shape (num_faces, ngon) with dtype matching faces.
            For dynamic-size: OffsetBlockedArray.
            Entry [i, j] is the face adjacent to face i across edge j.
        """
        if self._is_dynamic:
            wrapper = self._wrapper.manifold_edge_link_array()
            return OffsetBlockedArray(wrapper.offsets_array(), wrapper.data_array())
        else:
            return self._wrapper.manifold_edge_link_array()

    @manifold_edge_link.setter
    def manifold_edge_link(self, value: "np.ndarray | OffsetBlockedArray") -> None:
        """
        Set the manifold edge link array.

        Parameters
        ----------
        value : np.ndarray or OffsetBlockedArray
            Manifold edge link. Type must match mesh type (fixed vs dynamic).
        """
        if self._is_dynamic:
            if not isinstance(value, OffsetBlockedArray):
                raise TypeError(
                    "manifold_edge_link must be OffsetBlockedArray for dynamic mesh"
                )
            self._wrapper.set_manifold_edge_link(value._wrapper)
        else:
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    "manifold_edge_link must be numpy array for fixed-size mesh"
                )
            if not value.flags["C_CONTIGUOUS"]:
                value = np.ascontiguousarray(value)
            self._wrapper.set_manifold_edge_link(value)

    @property
    def face_link(self):
        """
        Get the face link structure.

        For each face, contains all faces connected to it by an edge.

        Builds the structure if not already built (also builds face_membership if needed).

        Returns
        -------
        OffsetBlockedArray
            Face link mapping faces to connected faces.
        """
        wrapper = self._wrapper.face_link_array()
        return OffsetBlockedArray(wrapper.offsets_array(), wrapper.data_array())

    @face_link.setter
    def face_link(self, value: OffsetBlockedArray) -> None:
        """
        Set the face link structure.

        Parameters
        ----------
        value : OffsetBlockedArray
            Face link structure.
        """
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
    def vertex_link(self, value: OffsetBlockedArray) -> None:
        """
        Set the vertex link structure.

        Parameters
        ----------
        value : OffsetBlockedArray
            Vertex link structure.
        """
        self._wrapper.set_vertex_link(value._wrapper)

    def build_normals(self) -> None:
        """
        Build the face normals.

        Builds the normals if not already built or if data has been modified.

        Raises
        ------
        ValueError
            If mesh is 2D (normals only supported for 3D meshes).
        """
        if self.dims != 3:
            raise ValueError("Normals only supported for 3D meshes")
        self._wrapper.build_normals()

    def build_point_normals(self) -> None:
        """
        Build the point (vertex) normals.

        Builds the normals if not already built or if data has been modified.
        Also builds face_membership and normals if needed.

        Raises
        ------
        ValueError
            If mesh is 2D (normals only supported for 3D meshes).
        """
        if self.dims != 3:
            raise ValueError("Point normals only supported for 3D meshes")
        self._wrapper.build_point_normals()

    @property
    def normals(self) -> np.ndarray:
        """
        Get the face normals.

        Returns unit normal vectors for each face. Computed by cross product
        of edge vectors.

        Builds the normals if not already built.

        Returns
        -------
        np.ndarray
            Array of shape (num_faces, 3) with unit normal vectors.

        Raises
        ------
        ValueError
            If mesh is 2D (normals only supported for 3D meshes).
        """
        if self.dims != 3:
            raise ValueError("Normals only supported for 3D meshes")
        return self._wrapper.normals_array()

    @normals.setter
    def normals(self, value: np.ndarray) -> None:
        """
        Set the face normals.

        Parameters
        ----------
        value : np.ndarray
            Array of shape (num_faces, 3) with unit normal vectors.

        Raises
        ------
        ValueError
            If mesh is 2D (normals only supported for 3D meshes).
        """
        if self.dims != 3:
            raise ValueError("Normals only supported for 3D meshes")
        if not value.flags["C_CONTIGUOUS"]:
            value = np.ascontiguousarray(value)
        self._wrapper.set_normals(value)

    @property
    def point_normals(self) -> np.ndarray:
        """
        Get the point (vertex) normals.

        Returns unit normal vectors for each vertex, computed by averaging
        the normals of adjacent faces.

        Builds the normals if not already built (also builds face_membership
        and face normals if needed).

        Returns
        -------
        np.ndarray
            Array of shape (num_points, 3) with unit normal vectors.

        Raises
        ------
        ValueError
            If mesh is 2D (normals only supported for 3D meshes).
        """
        if self.dims != 3:
            raise ValueError("Point normals only supported for 3D meshes")
        return self._wrapper.point_normals_array()

    @point_normals.setter
    def point_normals(self, value: np.ndarray) -> None:
        """
        Set the point (vertex) normals.

        Parameters
        ----------
        value : np.ndarray
            Array of shape (num_points, 3) with unit normal vectors.

        Raises
        ------
        ValueError
            If mesh is 2D (normals only supported for 3D meshes).
        """
        if self.dims != 3:
            raise ValueError("Point normals only supported for 3D meshes")
        if not value.flags["C_CONTIGUOUS"]:
            value = np.ascontiguousarray(value)
        self._wrapper.set_point_normals(value)

    def shared_view(self) -> "Mesh":
        """
        Create a new Mesh instance sharing the same underlying data.

        The new mesh shares the same faces, points, and cached structures (tree,
        face_membership, manifold_edge_link, etc.) but has its own transformation.
        This is useful when you need multiple mesh instances with different
        transformations but the same geometry.

        Returns
        -------
        Mesh
            New mesh instance sharing the same data, without transformation.

        Examples
        --------
        >>> mesh = tf.Mesh(faces, points)
        >>> mesh.transformation = transform_A
        >>> mesh2 = mesh.shared_view()
        >>> mesh2.transformation = transform_B
        >>> # mesh and mesh2 share the same data but have different transforms
        """
        new_mesh = object.__new__(Mesh)
        new_mesh._faces = self._faces
        new_mesh._points = self._points
        new_mesh._is_dynamic = self._is_dynamic
        new_mesh._wrapper = self._wrapper.shared_view()
        return new_mesh

    def __repr__(self) -> str:
        """String representation of the mesh."""
        if self._is_dynamic:
            return f"Mesh({self.number_of_points} points, {self.number_of_faces} faces, dynamic, {self.dims}D, dtype={self.dtype})"
        else:
            return f"Mesh({self.number_of_points} points, {self.number_of_faces} faces, {self.ngon}-gon, {self.dims}D, dtype={self.dtype})"
