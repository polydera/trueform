"""
Plane primitive

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

import numpy as np
from typing import Optional


class Plane:
    """
    An infinite plane in 2D or 3D space.

    In 2D: ax + by + c = 0
    In 3D: ax + by + cz + d = 0

    Parameters
    ----------
    coeffs : np.ndarray, optional
        Plane coefficients:
        - 2D: (a, b, c) for ax + by + c = 0, shape (3,)
        - 3D: (a, b, c, d) for ax + by + cz + d = 0, shape (4,)
    normal : np.ndarray, optional
        Normal vector, shape (D,). Must provide with origin.
    origin : np.ndarray, optional
        Point on plane, shape (D,). Must provide with normal.

    Examples
    --------
    >>> import numpy as np
    >>> from trueform import Plane
    >>> # Using coefficients
    >>> plane = Plane(coeffs=[0, 0, 1, -5])  # z = 5 plane
    >>> # Using normal and origin
    >>> plane = Plane(normal=[0, 0, 1], origin=[0, 0, 5])  # z = 5 plane
    >>> plane.normal
    array([0., 0., 1.], dtype=float32)
    >>> plane.offset
    -5.0
    """

    def __init__(self,
                 coeffs: Optional[np.ndarray] = None,
                 normal: Optional[np.ndarray] = None,
                 origin: Optional[np.ndarray] = None):

        if coeffs is not None:
            # Use coefficients directly and normalize
            coeffs = np.asarray(coeffs)

            if coeffs.ndim != 1:
                raise ValueError(f"Plane coefficients must be 1D array, got shape {coeffs.shape}")

            if coeffs.shape[0] not in [3, 4]:
                raise ValueError(f"Plane coefficients must have 3 (2D) or 4 (3D) elements, got {coeffs.shape[0]}")

            dims = coeffs.shape[0] - 1  # 2D or 3D

            # Normalize the normal vector and scale offset accordingly
            normal = coeffs[:-1]
            offset = coeffs[-1]
            normal_norm = np.linalg.norm(normal)

            if normal_norm < 1e-10:
                raise ValueError("Plane normal vector cannot be zero")

            # Normalize: divide both normal and offset by the norm
            normalized_normal = normal / normal_norm
            normalized_offset = offset / normal_norm

            data = np.concatenate([normalized_normal, [normalized_offset]])

        elif normal is not None and origin is not None:
            # Construct from normal and origin
            normal_arr = np.asarray(normal)
            origin_arr = np.asarray(origin)

            if normal_arr.shape != origin_arr.shape:
                raise ValueError(f"Normal and origin must have same shape, got {normal_arr.shape} and {origin_arr.shape}")

            if normal_arr.ndim != 1:
                raise ValueError(f"Normal and origin must be 1D arrays, got shape {normal_arr.shape}")

            dims = normal_arr.shape[0]
            if dims not in [2, 3]:
                raise ValueError(f"Plane must be 2D or 3D, got {dims} dimensions")

            # Normalize the normal vector
            normal_norm = np.linalg.norm(normal_arr)
            if normal_norm < 1e-10:
                raise ValueError("Plane normal vector cannot be zero")

            normalized_normal = normal_arr / normal_norm

            # Compute offset: d = -dot(normalized_normal, origin)
            offset = -np.dot(normalized_normal, origin_arr)

            # Store as [*normalized_normal, offset]
            data = np.concatenate([normalized_normal, [offset]])

        else:
            raise ValueError("Must provide either 'coeffs' or both 'normal' and 'origin'")

        # Validate dtype
        if data.dtype not in [np.float32, np.float64]:
            # Try to convert
            data = data.astype(np.float32)

        # Ensure C-contiguous
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)

        self._data = data
        self._dims = dims
        self._dtype = data.dtype

    @property
    def coeffs(self) -> np.ndarray:
        """Get plane coefficients (a, b, c[, d])."""
        return self._data

    @property
    def normal(self) -> np.ndarray:
        """Get normalized normal vector (unit length)."""
        return self._data[:-1]

    @property
    def offset(self) -> float:
        """Get offset coefficient (c in 2D, d in 3D)."""
        return float(self._data[-1])

    @property
    def data(self) -> np.ndarray:
        """Get underlying data array."""
        return self._data

    @property
    def dims(self) -> int:
        """Get dimensionality (2 or 3)."""
        return self._dims

    @property
    def dtype(self) -> np.dtype:
        """Get data type (float32 or float64)."""
        return self._dtype

    def __repr__(self) -> str:
        return f"Plane(coeffs={self._data.tolist()}, {self._dims}D, dtype={self._dtype})"
