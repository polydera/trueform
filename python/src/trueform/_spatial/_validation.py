"""
Validation utilities for spatial forms.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Tuple


def ensure_contiguous(arr: np.ndarray) -> np.ndarray:
    """Ensure array is C-contiguous, copying if necessary."""
    if not arr.flags["C_CONTIGUOUS"]:
        return np.ascontiguousarray(arr)
    return arr


def validate_points(
    points: np.ndarray,
    name: str = "points"
) -> Tuple[np.ndarray, int]:
    """
    Validate and normalize a points array.

    Parameters
    ----------
    points : np.ndarray
        Points array to validate
    name : str
        Name for error messages (default: "points")

    Returns
    -------
    points : np.ndarray
        Validated, C-contiguous points array (possibly converted to float32)
    dims : int
        Dimensionality (2 or 3)

    Raises
    ------
    TypeError
        If points is not a numpy array
    ValueError
        If points is not 2D or has invalid dimensionality
    """
    if not isinstance(points, np.ndarray):
        raise TypeError(f"Expected numpy array for {name}, got {type(points)}")
    if points.ndim != 2:
        raise ValueError(f"Expected 2D array for {name}, got shape {points.shape}")

    # Convert dtype if needed
    if points.dtype not in [np.float32, np.float64]:
        points = points.astype(np.float32)

    # Validate dimensionality
    dims = points.shape[1]
    if dims not in [2, 3]:
        raise ValueError(f"Points must be 2D or 3D, got {dims} dimensions")

    # Ensure C-contiguous
    points = ensure_contiguous(points)

    return points, dims


def validate_points_update(
    value: np.ndarray,
    original_dtype: np.dtype,
    original_dims: int,
    name: str = "Points"
) -> np.ndarray:
    """
    Validate a points array update (for setters).

    Parameters
    ----------
    value : np.ndarray
        New points array
    original_dtype : np.dtype
        Original dtype that must be matched
    original_dims : int
        Original dimensionality that must be matched
    name : str
        Name for error messages

    Returns
    -------
    np.ndarray
        Validated, C-contiguous array
    """
    if value.dtype != original_dtype:
        raise TypeError(
            f"{name} dtype ({value.dtype}) must match original dtype ({original_dtype})"
        )
    if value.shape[1] != original_dims:
        raise ValueError(
            f"{name} dimensionality ({value.shape[1]}) must match original ({original_dims})"
        )
    return ensure_contiguous(value)


def validate_index_array(
    indices: np.ndarray,
    expected_cols: int,
    name: str = "indices"
) -> np.ndarray:
    """
    Validate an index array (faces, edges).

    Parameters
    ----------
    indices : np.ndarray
        Index array to validate
    expected_cols : int
        Expected number of columns (2 for edges, 3 for triangles)
    name : str
        Name for error messages

    Returns
    -------
    np.ndarray
        Validated, C-contiguous array

    Raises
    ------
    TypeError
        If not a numpy array or has invalid dtype
    ValueError
        If shape is invalid
    """
    if not isinstance(indices, np.ndarray):
        raise TypeError(f"Expected numpy array for {name}, got {type(indices)}")
    if indices.ndim != 2:
        raise ValueError(f"Expected 2D array for {name}, got shape {indices.shape}")
    if indices.shape[1] != expected_cols:
        raise ValueError(
            f"{name.capitalize()} must have {expected_cols} vertices, got {indices.shape[1]}"
        )
    if indices.dtype not in [np.int32, np.int64]:
        raise TypeError(
            f"{name.capitalize()} must be int32 or int64, got {indices.dtype}. "
            f"Convert with {name}.astype(np.int32) or {name}.astype(np.int64)"
        )
    return ensure_contiguous(indices)


def validate_index_update(
    value: np.ndarray,
    original_dtype: np.dtype,
    expected_cols: int,
    name: str = "Indices"
) -> np.ndarray:
    """
    Validate an index array update (for setters).

    Parameters
    ----------
    value : np.ndarray
        New index array
    original_dtype : np.dtype
        Original dtype that must be matched
    expected_cols : int
        Expected number of columns
    name : str
        Name for error messages

    Returns
    -------
    np.ndarray
        Validated, C-contiguous array
    """
    if value.dtype != original_dtype:
        raise TypeError(
            f"{name} dtype ({value.dtype}) must match original dtype ({original_dtype})"
        )
    if value.shape[1] != expected_cols:
        raise ValueError(
            f"{name} must have {expected_cols} vertices, got {value.shape[1]}"
        )
    return ensure_contiguous(value)


def validate_transformation(
    mat: np.ndarray,
    dims: int,
    points_dtype: np.dtype
) -> np.ndarray:
    """
    Validate a transformation matrix.

    Parameters
    ----------
    mat : np.ndarray
        Transformation matrix to validate
    dims : int
        Point dimensionality (2 or 3)
    points_dtype : np.dtype
        Points dtype that transformation must match

    Returns
    -------
    np.ndarray
        Validated, C-contiguous transformation matrix
    """
    expected_size = dims + 1
    if mat.shape != (expected_size, expected_size):
        raise ValueError(
            f"Transformation must be {expected_size}x{expected_size} for {dims}D points, "
            f"got shape {mat.shape}"
        )
    if mat.dtype != points_dtype:
        raise TypeError(
            f"Transformation dtype ({mat.dtype}) must match points dtype ({points_dtype})"
        )
    return ensure_contiguous(mat)
