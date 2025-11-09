"""
Point primitive

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
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

    def __repr__(self) -> str:
        return f"Point({self._data.tolist()}, dtype={self._dtype})"
