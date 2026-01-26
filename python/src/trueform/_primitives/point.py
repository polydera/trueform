"""
Point primitive

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np


class Point:
    """
    A point in 2D or 3D space.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates, shape (D,) where D is 2 or 3, dtype float32 or float64

    Examples
    --------
    >>> import numpy as np
    >>> from trueform import Point
    >>> p = Point([1.0, 2.0, 3.0])
    >>> p.dims
    3
    >>> p.coords
    array([1., 2., 3.], dtype=float32)
    """

    def __init__(self, coords: np.ndarray):
        # Convert to numpy if needed
        coords = np.asarray(coords)

        # Validate shape
        if coords.ndim != 1:
            raise ValueError(f"Point must be 1D array, got shape {coords.shape}")

        # Validate dimensionality
        if coords.shape[0] not in [2, 3]:
            raise ValueError(f"Point must be 2D or 3D, got {coords.shape[0]} dimensions")

        # Validate dtype
        if coords.dtype not in [np.float32, np.float64]:
            # Try to convert
            coords = coords.astype(np.float32)

        # Ensure C-contiguous
        if not coords.flags['C_CONTIGUOUS']:
            coords = np.ascontiguousarray(coords)

        self._data = coords
        self._dims = coords.shape[0]
        self._dtype = coords.dtype

    @property
    def coords(self) -> np.ndarray:
        """Get point coordinates."""
        return self._data

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

    @property
    def x(self) -> float:
        """Get x coordinate."""
        return float(self._data[0])

    @property
    def y(self) -> float:
        """Get y coordinate."""
        return float(self._data[1])

    @property
    def z(self) -> float:
        """
        Get z coordinate.

        For 2D points, returns 0.0.

        Returns
        -------
        float
            Z coordinate, or 0.0 if point is 2D
        """
        if self._dims < 3:
            return 0.0
        return float(self._data[2])

    @classmethod
    def from_xy(cls, x: float, y: float, dtype=np.float32):
        """
        Create a 2D point from x and y coordinates.

        Parameters
        ----------
        x : float
            X coordinate
        y : float
            Y coordinate
        dtype : numpy dtype, optional
            Data type for coordinates (float32 or float64). Default is float32.

        Returns
        -------
        Point
            2D point at (x, y)

        Examples
        --------
        >>> pt = Point.from_xy(1.0, 2.0)
        >>> pt.coords
        array([1., 2.], dtype=float32)
        """
        return cls(np.array([x, y], dtype=dtype))

    @classmethod
    def from_xyz(cls, x: float, y: float, z: float, dtype=np.float32):
        """
        Create a 3D point from x, y, and z coordinates.

        Parameters
        ----------
        x : float
            X coordinate
        y : float
            Y coordinate
        z : float
            Z coordinate
        dtype : numpy dtype, optional
            Data type for coordinates (float32 or float64). Default is float32.

        Returns
        -------
        Point
            3D point at (x, y, z)

        Examples
        --------
        >>> pt = Point.from_xyz(1.0, 2.0, 3.0)
        >>> pt.coords
        array([1., 2., 3.], dtype=float32)
        """
        return cls(np.array([x, y, z], dtype=dtype))

    def __repr__(self) -> str:
        return f"Point({self._data.tolist()}, dtype={self._dtype})"
