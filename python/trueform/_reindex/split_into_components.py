"""
split_into_components() function implementation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

import numpy as np
from typing import Union, Tuple, List
from .. import _trueform
from .._spatial import Mesh, EdgeMesh


def split_into_components(
    data: Union[Tuple[np.ndarray, np.ndarray], Mesh, EdgeMesh],
    labels: np.ndarray
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """
    Split geometry into separate components based on labels.

    Groups primitives (triangles or quads) by their label values and creates
    separate geometries for each unique label. Primitives with the same label
    are grouped together, and unused points are automatically filtered.

    Parameters
    ----------
    data : tuple, Mesh, or EdgeMesh
        Input geometric data:
        - Indexed geometry: tuple (indices, points) where:
          * indices: shape (N, V) with dtype int32 or int64, V = 2, 3, or 4
          * points: shape (M, Dims) where Dims = 2 or 3
        - Mesh: tf.Mesh object (2D or 3D, NGon = 3 or 4)
        - EdgeMesh: tf.EdgeMesh object (2D or 3D)
    labels : np.ndarray
        1D array of labels, one per primitive, shape (N,) with dtype int32.
        Primitives with the same label value will be grouped into the same component.

    Returns
    -------
    components : list of tuple
        List of (connectivity, points) tuples, one per unique label.
        Each component contains the primitives sharing that label,
        with points reindexed and unused points removed.
    component_labels : np.ndarray
        1D array of label values corresponding to each component, shape (K,).
        component_labels[i] is the label value for components[i].

    Raises
    ------
    TypeError
        If data is not a tuple, Mesh, or EdgeMesh
        If labels dtype is not int32
        If indices/points have incorrect dtypes
    ValueError
        If arrays have incorrect shapes or dimensions

    Notes
    -----
    - Supports edges (V=2), triangles (V=3), and quads (V=4)
    - PointCloud is NOT supported (no indexed geometry)
    - Components are ordered by label value (sorted)
    - Each component is a standalone geometry with reindexed vertices

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Create mesh with labeled faces
    >>> faces = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4], [4, 3, 5]], dtype=np.int32)
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0],
    ...                    [1.5, 1, 0], [1, 2, 0], [2, 2, 0]], dtype=np.float32)
    >>> labels = np.array([0, 0, 1, 1], dtype=np.int32)  # Two components
    >>>
    >>> # Split into components
    >>> components, comp_labels = tf.split_into_components((faces, points), labels)
    >>> print(f"Found {len(components)} components")
    Found 2 components
    >>>
    >>> # Access individual components
    >>> comp0_faces, comp0_points = components[0]
    >>> print(f"Component 0 (label={comp_labels[0]}): {comp0_faces.shape[0]} faces")
    Component 0 (label=0): 2 faces
    >>>
    >>> # Use with Mesh object
    >>> mesh = tf.Mesh(faces, points)
    >>> components, comp_labels = tf.split_into_components(mesh, labels)
    >>>
    >>> # Process each component separately
    >>> for i, ((comp_faces, comp_points), label) in enumerate(zip(components, comp_labels)):
    ...     print(f"Component {i}: label={label}, {len(comp_faces)} faces, {len(comp_points)} points")
    Component 0: label=0, 2 faces, 4 points
    Component 1: label=1, 2 faces, 4 points
    """

    # Validate labels array
    if not isinstance(labels, np.ndarray):
        raise TypeError(
            f"labels must be np.ndarray, got {type(labels).__name__}"
        )

    if labels.ndim != 1:
        raise ValueError(
            f"labels must be 1D array with shape (N,), got shape {labels.shape}"
        )

    if labels.dtype != np.int32:
        raise TypeError(
            f"labels dtype must be int32, got {labels.dtype}. "
            f"Convert with labels.astype(np.int32)"
        )

    # Ensure C-contiguous
    if not labels.flags['C_CONTIGUOUS']:
        labels = np.ascontiguousarray(labels)

    # ===== HANDLE TUPLE INPUT (INDEXED GEOMETRY) =====
    if isinstance(data, tuple):
        if len(data) != 2:
            raise ValueError(
                f"Tuple input must have exactly 2 elements (indices, points), got {len(data)}"
            )

        indices, points = data

        # Validate indices
        if not isinstance(indices, np.ndarray):
            raise TypeError(
                f"indices must be np.ndarray, got {type(indices).__name__}"
            )

        if indices.ndim != 2:
            raise ValueError(
                f"indices must be 2D array with shape (N, V), got shape {indices.shape}"
            )

        if indices.dtype not in (np.int32, np.int64):
            raise TypeError(
                f"indices dtype must be int32 or int64, got {indices.dtype}. "
                f"Convert with indices.astype(np.int32) or indices.astype(np.int64)"
            )

        # Validate points
        if not isinstance(points, np.ndarray):
            raise TypeError(
                f"points must be np.ndarray, got {type(points).__name__}"
            )

        if points.ndim != 2:
            raise ValueError(
                f"points must be 2D array with shape (M, Dims), got shape {points.shape}"
            )

        if points.dtype not in (np.float32, np.float64):
            raise TypeError(
                f"points dtype must be float32 or float64, got {points.dtype}. "
                f"Convert with points.astype(np.float32) or points.astype(np.float64)"
            )

        # Extract shape information
        V = indices.shape[1]
        dims = points.shape[1]

        # Validate V
        if V not in (2, 3, 4):
            raise ValueError(
                f"split_into_components supports edges (V=2), triangles (V=3), or quads (V=4), got V={V}"
            )

        # Validate dims
        if dims not in (2, 3):
            raise ValueError(
                f"points must have 2 or 3 dimensions, got dims={dims}"
            )

        # Validate size match
        if indices.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Number of labels ({labels.shape[0]}) must match number of primitives ({indices.shape[0]})"
            )

        # Ensure C-contiguous
        if not indices.flags['C_CONTIGUOUS']:
            indices = np.ascontiguousarray(indices)
        if not points.flags['C_CONTIGUOUS']:
            points = np.ascontiguousarray(points)

        # Build suffix: {V}{index}{real}{dims}d
        index_str = 'int' if indices.dtype == np.int32 else 'int64'
        real_str = 'float' if points.dtype == np.float32 else 'double'
        suffix = f"{V}{index_str}{real_str}{dims}d"

        # Build function name
        func_name = f"split_into_components_{suffix}"

        # Call C++ function - returns (list_of_components, component_labels)
        cpp_func = getattr(_trueform.reindex, func_name)
        components, comp_labels = cpp_func(indices, points, labels)

        return components, comp_labels

    # ===== HANDLE FORM OBJECTS (Mesh, EdgeMesh) =====
    elif isinstance(data, (Mesh, EdgeMesh)):
        # Validate dims
        if data.dims not in (2, 3):
            raise ValueError(
                f"{type(data).__name__} dims must be 2 or 3, got {data.dims}D"
            )

        if isinstance(data, Mesh):
            # Extract arrays from Mesh
            indices = data.faces
            points = data.points
            V = data.ngon

            # Validate V (only triangles and quads for Mesh)
            if V not in (3, 4):
                raise ValueError(
                    f"split_into_components only supports triangular (NGon=3) or quad (NGon=4) meshes, got NGon={V}"
                )

            element_name = "faces"

        else:  # EdgeMesh
            # Extract arrays from EdgeMesh
            indices = data.edges
            points = data.points
            V = 2
            element_name = "edges"

        dims = data.dims

        # Validate size match
        if indices.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Number of labels ({labels.shape[0]}) must match number of {element_name} ({indices.shape[0]})"
            )

        # Build suffix: {V}{index}{real}{dims}d
        index_str = 'int' if indices.dtype == np.int32 else 'int64'
        real_str = 'float' if points.dtype == np.float32 else 'double'
        suffix = f"{V}{index_str}{real_str}{dims}d"

        # Build function name
        func_name = f"split_into_components_{suffix}"

        # Call C++ function - returns (list_of_components, component_labels)
        cpp_func = getattr(_trueform.reindex, func_name)
        components, comp_labels = cpp_func(indices, points, labels)

        return components, comp_labels

    else:
        raise TypeError(
            f"Expected tuple (indices, points), Mesh, or EdgeMesh, got {type(data).__name__}. "
            f"Note: PointCloud is not supported by split_into_components."
        )
