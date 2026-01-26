"""
Plane primitive

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Optional


class Plane:
    """
    An infinite plane in 3D space.

    Equation: ax + by + cz + d = 0

    Parameters
    ----------
    coeffs : np.ndarray, optional
        Plane coefficients (a, b, c, d) for ax + by + cz + d = 0, shape (4,)
    normal : np.ndarray, optional
        Normal vector, shape (3,). Must provide with origin.
    origin : np.ndarray, optional
        Point on plane, shape (3,). Must provide with normal.

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

            if coeffs.shape[0] != 4:
                raise ValueError(f"Plane coefficients must have 4 elements (a, b, c, d), got {coeffs.shape[0]}")

            dims = 3  # Plane is 3D only

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
            if dims != 3:
                raise ValueError(f"Plane must be 3D, got {dims} dimensions. Normal and origin must have shape (3,)")

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
        """Get offset coefficient d."""
        return float(self._data[-1])

    @property
    def data(self) -> np.ndarray:
        """Get underlying data array."""
        return self._data

    @property
    def dims(self) -> int:
        """Get dimensionality (always 3 for Plane)."""
        return self._dims

    @property
    def dtype(self) -> np.dtype:
        """Get data type (float32 or float64)."""
        return self._dtype

    @classmethod
    def from_point_normal(cls, origin, normal):
        """
        Create plane from a point and normal vector.

        This is equivalent to Plane(normal=..., origin=...) but as a classmethod
        for consistency with other primitives' from_* constructors.

        Parameters
        ----------
        origin : array-like
            Point on the plane, shape (3,)
        normal : array-like
            Normal vector to the plane, shape (3,)

        Returns
        -------
        Plane
            Plane passing through origin with given normal

        Examples
        --------
        >>> plane = Plane.from_point_normal([0, 0, 5], [0, 0, 1])
        >>> plane.normal
        array([0., 0., 1.], dtype=float32)
        >>> plane.offset
        -5.0
        """
        return cls(normal=normal, origin=origin)

    @classmethod
    def from_points(cls, p1, p2, p3):
        """
        Create plane passing through three points.

        The normal vector is computed as (p2 - p1) × (p3 - p1).

        Parameters
        ----------
        p1 : array-like
            First point on the plane, shape (3,)
        p2 : array-like
            Second point on the plane, shape (3,)
        p3 : array-like
            Third point on the plane, shape (3,)

        Returns
        -------
        Plane
            Plane passing through the three points

        Raises
        ------
        ValueError
            If points are collinear (don't define a unique plane)

        Examples
        --------
        >>> plane = Plane.from_points([0, 0, 0], [1, 0, 0], [0, 1, 0])
        >>> plane.normal
        array([0., 0., 1.], dtype=float32)
        """
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        p3 = np.asarray(p3)

        if p1.shape[0] != 3 or p2.shape[0] != 3 or p3.shape[0] != 3:
            raise ValueError(f"Plane.from_points requires 3D points, got shapes {p1.shape}, {p2.shape}, {p3.shape}")

        # Compute normal via cross product
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)

        # Check if points are collinear
        if np.linalg.norm(normal) < 1e-10:
            raise ValueError("Points are collinear and do not define a unique plane")

        return cls(normal=normal, origin=p1)

    def __repr__(self) -> str:
        return f"Plane(coeffs={self._data.tolist()}, {self._dims}D, dtype={self._dtype})"
