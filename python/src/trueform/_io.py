"""
IO utilities for reading and writing mesh files

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Tuple, Optional, Union
from . import _trueform
from ._spatial.mesh import Mesh
from ._dispatch import InputMeta, build_suffix


def read_stl(filename: str, index_dtype: Union[type, np.dtype] = np.int32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read an STL file and return mesh data as numpy arrays.

    Parameters
    ----------
    filename : str
        Path to the STL file to read
    index_dtype : dtype, optional
        Data type for face indices. Must be np.int32 or np.int64. Default is np.int32.

    Returns
    -------
    faces : ndarray of shape (num_faces, 3) with dtype specified by index_dtype
        Face indices into the points array. Each row contains three indices
        that reference vertices in the points array, forming a triangle.
    points : ndarray of shape (num_points, 3) and dtype float32
        3D coordinates of mesh vertices. Each row is a (x, y, z) coordinate.

    Examples
    --------
    >>> import trueform as tf
    >>> # Read with default int32 indices
    >>> faces, points = tf.read_stl("model.stl")
    >>> print(f"Faces dtype: {faces.dtype}")  # int32
    >>>
    >>> # Read with int64 indices for large meshes
    >>> faces, points = tf.read_stl("model.stl", index_dtype=np.int64)
    >>> print(f"Faces dtype: {faces.dtype}")  # int64

    Notes
    -----
    The STL file format stores triangular mesh data. This function:
    - Reads both ASCII and binary STL files
    - Cleans duplicate vertices (merges points that are identical)
    - Returns zero-copy numpy arrays (memory is managed by numpy)
    - Face indices are 0-based
    """
    # Normalize dtype
    if isinstance(index_dtype, type):
        index_dtype = np.dtype(index_dtype)
    elif not isinstance(index_dtype, np.dtype):
        index_dtype = np.dtype(index_dtype)

    # Validate and dispatch
    if index_dtype == np.int32:
        return _trueform.io.read_stl_int32(filename)
    elif index_dtype == np.int64:
        return _trueform.io.read_stl_int64(filename)
    else:
        raise ValueError(
            f"index_dtype must be np.int32 or np.int64, got {index_dtype}"
        )


def write_stl(
    data: Union[Tuple[np.ndarray, np.ndarray], Mesh],
    filename: str,
    transformation: Optional[np.ndarray] = None
) -> bool:
    """
    Write a triangular mesh to an STL file.

    Supports indexed geometry as tuples or Mesh objects:
    - Tuple: write_stl((faces, points), filename, transformation=None)
    - Mesh: write_stl(mesh, filename, transformation=None)

    Parameters
    ----------
    data : tuple or Mesh
        Input geometric data:
        - Tuple (faces, points) where:
          * faces: shape (N, 3) with dtype int32 or int64
          * points: shape (M, 3) with dtype float32
        - Mesh object (must be 3D triangular mesh)
    filename : str
        Path to output STL file.
        The .stl extension will be appended if not present.
    transformation : ndarray of shape (4, 4) with dtype float32, optional
        Homogeneous transformation matrix to apply before writing.
        If provided, overrides any transformation set on the Mesh object.

    Returns
    -------
    success : bool
        True if the file was written successfully, False otherwise.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Tuple input
    >>> faces = np.array([[0, 1, 2]], dtype=np.int32)
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    >>> tf.write_stl((faces, points), "triangle.stl")
    >>>
    >>> # Tuple input with transformation
    >>> transform = np.eye(4, dtype=np.float32)
    >>> transform[2, 3] = 5.0  # Translate 5 units in Z
    >>> tf.write_stl((faces, points), "triangle_translated.stl", transformation=transform)
    >>>
    >>> # Mesh object
    >>> mesh = tf.Mesh(faces, points)
    >>> tf.write_stl(mesh, "triangle.stl")
    >>>
    >>> # Mesh with transformation
    >>> mesh.transformation = transform
    >>> tf.write_stl(mesh, "triangle_transformed.stl")
    >>>
    >>> # Override mesh transformation
    >>> tf.write_stl(mesh, "triangle_custom.stl", transformation=other_transform)

    Notes
    -----
    - The STL format only supports triangular meshes. All faces must have exactly 3 vertices.
    - For Mesh objects, only triangle meshes (ngon=3) are supported, not quads.
    - Points must be 3D (shape (N, 3))
    - Binary STL format is used for writing
    - Arrays must be C-contiguous
    - When using a Mesh object, if the explicit transformation kwarg is provided,
      it overrides the mesh's transformation property
    """
    # Handle Mesh object
    if isinstance(data, Mesh):
        mesh = data

        # Validate Mesh requirements for STL
        if mesh.dims != 3:
            raise ValueError(
                f"STL format only supports 3D meshes, but mesh has {mesh.dims}D points. "
                f"Only 3D meshes can be written to STL."
            )

        if mesh.is_dynamic or mesh.ngon != 3:
            raise ValueError(
                f"STL format only supports triangular meshes, but mesh has "
                f"{'dynamic' if mesh.is_dynamic else f'{mesh.ngon}-gon'} faces. "
                f"Convert to triangles first."
            )

        # Extract data from mesh
        faces = mesh.faces
        points = mesh.points

        # Use mesh transformation if no explicit override
        if transformation is None and mesh.transformation is not None:
            transformation = mesh.transformation

    # Handle tuple input
    elif isinstance(data, tuple):
        if len(data) != 2:
            raise ValueError(
                f"Tuple input must have exactly 2 elements (faces, points), got {len(data)}"
            )

        faces, points = data

    else:
        raise TypeError(
            f"Expected tuple or Mesh object, got {type(data).__name__}"
        )

    # Validate faces shape and dtype
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"faces must have shape (N, 3), got shape {faces.shape}"
        )

    faces_dtype = faces.dtype
    if faces_dtype not in (np.int32, np.int64):
        raise ValueError(
            f"faces dtype must be np.int32 or np.int64, got {faces_dtype}"
        )

    # Validate points shape and dtype
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points must have shape (M, 3), got shape {points.shape}"
        )

    if points.dtype != np.float32:
        raise ValueError(
            f"points dtype must be np.float32, got {points.dtype}"
        )

    # Validate transformation if provided
    if transformation is not None:
        if transformation.shape != (4, 4):
            raise ValueError(
                f"transformation must have shape (4, 4), got {transformation.shape}"
            )
        if transformation.dtype != np.float32:
            raise ValueError(
                f"transformation dtype must be np.float32, got {transformation.dtype}"
            )
        if not transformation.flags['C_CONTIGUOUS']:
            transformation = np.ascontiguousarray(transformation)

    # Ensure arrays are C-contiguous
    if not faces.flags['C_CONTIGUOUS']:
        faces = np.ascontiguousarray(faces)
    if not points.flags['C_CONTIGUOUS']:
        points = np.ascontiguousarray(points)

    # Dispatch based on faces dtype
    if faces_dtype == np.int32:
        return _trueform.io.write_stl_int32(faces, points, transformation, filename)
    else:  # int64
        return _trueform.io.write_stl_int64(faces, points, transformation, filename)


def read_obj(filename: str, ngon: int, index_dtype: Union[type, np.dtype] = np.int32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read an OBJ file and return mesh data as numpy arrays.

    Parameters
    ----------
    filename : str
        Path to the OBJ file to read
    ngon : int
        Number of vertices per face. Must be 3 (triangles) or 4 (quads).
        This parameter is required because numpy arrays cannot be ragged
        (all rows must have the same length). The returned faces array
        will have shape (num_faces, ngon).
    index_dtype : dtype, optional
        Data type for face indices. Must be np.int32 or np.int64. Default is np.int32.

    Returns
    -------
    faces : ndarray of shape (num_faces, ngon) with dtype specified by index_dtype
        Face indices into the points array. Each row contains ngon indices
        that reference vertices in the points array.
    points : ndarray of shape (num_points, 3) and dtype float32
        3D coordinates of mesh vertices. Each row is a (x, y, z) coordinate.

    Examples
    --------
    >>> import trueform as tf
    >>> # Read triangular mesh with default int32 indices
    >>> faces, points = tf.read_obj("model.obj", ngon=3)
    >>> print(f"Faces shape: {faces.shape}")  # (N, 3)
    >>>
    >>> # Read quad mesh with int64 indices
    >>> faces, points = tf.read_obj("quad_model.obj", ngon=4, index_dtype=np.int64)
    >>> print(f"Faces shape: {faces.shape}")  # (N, 4)

    Notes
    -----
    The OBJ file format stores indexed mesh data. This function:
    - Reads ASCII OBJ files
    - Returns zero-copy numpy arrays (memory is managed by numpy)
    - Face indices are 0-based (converted from OBJ's 1-based indexing)
    - Only vertex positions are read (normals and texture coordinates are ignored)
    - If the OBJ file contains faces with more vertices than ngon, only the
      first ngon vertices are kept; faces with fewer vertices cause an error
    """
    # Normalize dtype
    if isinstance(index_dtype, type):
        index_dtype = np.dtype(index_dtype)
    elif not isinstance(index_dtype, np.dtype):
        index_dtype = np.dtype(index_dtype)

    # Validate ngon
    if ngon not in (3, 4):
        raise ValueError(f"ngon must be 3 or 4, got {ngon}")

    # Validate index_dtype
    if index_dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
        raise ValueError(
            f"index_dtype must be np.int32 or np.int64, got {index_dtype}"
        )

    # Build suffix and dispatch
    index_str = 'int' if index_dtype == np.int32 else 'int64'
    suffix = f"{index_str}{ngon}"
    func = getattr(_trueform.io, f"read_obj_{suffix}")
    return func(filename)


def write_obj(
    data: Union[Tuple[np.ndarray, np.ndarray], Mesh],
    filename: str,
    transformation: Optional[np.ndarray] = None
) -> bool:
    """
    Write a mesh to an OBJ file.

    Supports indexed geometry as tuples or Mesh objects:
    - Tuple: write_obj((faces, points), filename, transformation=None)
    - Mesh: write_obj(mesh, filename, transformation=None)

    Parameters
    ----------
    data : tuple or Mesh
        Input geometric data:
        - Tuple (faces, points) where:
          * faces: shape (N, ngon) with dtype int32 or int64
          * points: shape (M, 3) with dtype float32 or float64
        - Mesh object (must be 3D mesh)
    filename : str
        Path to output OBJ file.
        The .obj extension will be appended if not present.
    transformation : ndarray of shape (4, 4) with dtype float32, optional
        Homogeneous transformation matrix to apply before writing.
        If provided, overrides any transformation set on the Mesh object.

    Returns
    -------
    success : bool
        True if the file was written successfully, False otherwise.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Tuple input - triangular mesh
    >>> faces = np.array([[0, 1, 2]], dtype=np.int32)
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    >>> tf.write_obj((faces, points), "triangle.obj")
    >>>
    >>> # Quad mesh
    >>> faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    >>> tf.write_obj((faces, points), "quad.obj")
    >>>
    >>> # Mesh object
    >>> mesh = tf.Mesh(faces, points)
    >>> tf.write_obj(mesh, "mesh.obj")

    Notes
    -----
    - Supports both triangular (ngon=3) and quad (ngon=4) meshes
    - Points must be 3D (shape (N, 3))
    - ASCII OBJ format is used for writing
    - Arrays must be C-contiguous
    """
    # Handle Mesh object
    if isinstance(data, Mesh):
        mesh = data

        # Validate Mesh requirements for OBJ
        if mesh.dims != 3:
            raise ValueError(
                f"OBJ format only supports 3D meshes, but mesh has {mesh.dims}D points."
            )

        if mesh.is_dynamic or mesh.ngon not in (3, 4):
            raise ValueError(
                f"OBJ format only supports triangular or quad meshes, but mesh has "
                f"{'dynamic' if mesh.is_dynamic else f'{mesh.ngon}-gon'} faces."
            )

        # Extract data from mesh
        faces = mesh.faces
        points = mesh.points

        # Use mesh transformation if no explicit override
        if transformation is None and mesh.transformation is not None:
            transformation = mesh.transformation

    # Handle tuple input
    elif isinstance(data, tuple):
        if len(data) != 2:
            raise ValueError(
                f"Tuple input must have exactly 2 elements (faces, points), got {len(data)}"
            )

        faces, points = data

    else:
        raise TypeError(
            f"Expected tuple or Mesh object, got {type(data).__name__}"
        )

    # Validate faces shape and dtype
    if faces.ndim != 2 or faces.shape[1] not in (3, 4):
        raise ValueError(
            f"faces must have shape (N, 3) or (N, 4), got shape {faces.shape}"
        )

    ngon = str(faces.shape[1])
    faces_dtype = faces.dtype
    if faces_dtype not in (np.int32, np.int64):
        raise ValueError(
            f"faces dtype must be np.int32 or np.int64, got {faces_dtype}"
        )

    # Validate points shape and dtype
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points must have shape (M, 3), got shape {points.shape}"
        )

    points_dtype = points.dtype
    if points_dtype not in (np.float32, np.float64):
        raise ValueError(
            f"points dtype must be np.float32 or np.float64, got {points_dtype}"
        )

    # Validate transformation if provided
    if transformation is not None:
        if transformation.shape != (4, 4):
            raise ValueError(
                f"transformation must have shape (4, 4), got {transformation.shape}"
            )
        if transformation.dtype != np.float32:
            raise ValueError(
                f"transformation dtype must be np.float32, got {transformation.dtype}"
            )
        if not transformation.flags['C_CONTIGUOUS']:
            transformation = np.ascontiguousarray(transformation)

    # Ensure arrays are C-contiguous
    if not faces.flags['C_CONTIGUOUS']:
        faces = np.ascontiguousarray(faces)
    if not points.flags['C_CONTIGUOUS']:
        points = np.ascontiguousarray(points)

    # Build suffix and dispatch
    meta = InputMeta(faces_dtype, points_dtype, ngon, 3)
    suffix = build_suffix(meta)
    func = getattr(_trueform.io, f"write_obj_{suffix}")
    return func(faces, points, transformation, filename)
