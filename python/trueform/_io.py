"""
IO utilities for reading and writing mesh files

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

import numpy as np
from typing import Tuple, Optional, Union
from . import _trueform
from ._core.mesh import Mesh


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
    faces_or_mesh: Union[np.ndarray, Mesh],
    points_or_filename: Union[np.ndarray, str, None] = None,
    filename: Optional[str] = None,
    transformation: Optional[np.ndarray] = None
) -> bool:
    """
    Write a triangular mesh to an STL file.

    Can be called with either raw arrays or a Mesh object:
    - Raw arrays: write_stl(faces, points, filename, transformation=None)
    - Mesh object: write_stl(mesh, filename, transformation=None)

    Parameters
    ----------
    faces_or_mesh : ndarray or Mesh
        Either a faces array of shape (num_faces, 3) with dtype int32 or int64,
        or a Mesh object.
    points_or_filename : ndarray or str, optional
        If faces_or_mesh is an array: points array of shape (num_points, 3) with dtype float32
        If faces_or_mesh is a Mesh: filename string
    filename : str, optional
        Path to output STL file (only used when passing raw arrays).
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
    >>> # Raw arrays (backward compatible)
    >>> faces = np.array([[0, 1, 2]], dtype=np.int32)
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    >>> tf.write_stl(faces, points, "triangle.stl")
    >>>
    >>> # Raw arrays with transformation
    >>> transform = np.eye(4, dtype=np.float32)
    >>> transform[2, 3] = 5.0  # Translate 5 units in Z
    >>> tf.write_stl(faces, points, "triangle_translated.stl", transformation=transform)
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
    # Detect if first argument is a Mesh object
    if isinstance(faces_or_mesh, Mesh):
        mesh = faces_or_mesh

        # Validate Mesh requirements for STL
        if mesh.dims != 3:
            raise ValueError(
                f"STL format only supports 3D meshes, but mesh has {mesh.dims}D points. "
                f"Only 3D meshes can be written to STL."
            )

        if mesh.ngon != 3:
            raise ValueError(
                f"STL format only supports triangular meshes, but mesh has {mesh.ngon}-gon faces. "
                f"Convert quad mesh to triangles first."
            )

        # Extract data from mesh
        faces = mesh.faces
        points = mesh.points

        # Determine filename (second argument when using Mesh)
        if isinstance(points_or_filename, str):
            actual_filename = points_or_filename
        elif filename is not None:
            actual_filename = filename
        else:
            raise ValueError(
                "When passing a Mesh object, provide filename as second argument: "
                "write_stl(mesh, 'file.stl')"
            )

        # Use mesh transformation if no explicit override
        if transformation is None and mesh.transformation is not None:
            transformation = mesh.transformation
    else:
        # Original array-based API
        faces = faces_or_mesh
        points = points_or_filename
        actual_filename = filename

        if points is None:
            raise ValueError(
                "When passing arrays, provide both faces and points: "
                "write_stl(faces, points, 'file.stl')"
            )

        if actual_filename is None:
            raise ValueError(
                "When passing arrays, provide filename as third argument: "
                "write_stl(faces, points, 'file.stl')"
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
        return _trueform.io.write_stl_int32(faces, points, transformation, actual_filename)
    else:  # int64
        return _trueform.io.write_stl_int64(faces, points, transformation, actual_filename)
