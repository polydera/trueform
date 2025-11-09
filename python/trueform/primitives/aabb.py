"""
AABB (Axis-Aligned Bounding Box) primitive

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

import numpy as np
from typing import Optional


class AABB:
    """
    An axis-aligned bounding box defined by min and max corners.

    Parameters
    ----------
    min : np.ndarray, optional
        Minimum corner, shape (D,) where D is 2 or 3
    max : np.ndarray, optional
        Maximum corner, shape (D,)
    bounds : np.ndarray, optional
        Alternative: shape (2, D) with [min, max]

    Examples
    --------
    >>> import numpy as np
    >>> from trueform import AABB
    >>> # Using min/max
    >>> box = AABB(min=[0, 0, 0], max=[1, 1, 1])
    >>> box.min
    array([0., 0., 0.], dtype=float32)
    >>> # Using bounds array
    >>> box = AABB(bounds=[[0, 0, 0], [1, 1, 1]])
    """

    def __init__(self,
                 min: Optional[np.ndarray] = None,
                 max: Optional[np.ndarray] = None,
                 bounds: Optional[np.ndarray] = None):

        if bounds is not None:
            # Use bounds array directly
            bounds = np.asarray(bounds)
            if bounds.shape[0] != 2:
                raise ValueError(f"AABB bounds must have shape (2, D), got {bounds.shape}")
            data = bounds
        elif min is not None and max is not None:
            # Construct from min/max
            min_arr = np.asarray(min)
            max_arr = np.asarray(max)

            if min_arr.shape != max_arr.shape:
                raise ValueError(f"Min and max must have same shape, got {min_arr.shape} and {max_arr.shape}")

            if min_arr.ndim != 1:
                raise ValueError(f"Min and max must be 1D arrays, got shape {min_arr.shape}")

            data = np.stack([min_arr, max_arr], axis=0)
        else:
            raise ValueError("Must provide either 'bounds' or both 'min' and 'max'")

        # Validate shape
        if data.ndim != 2 or data.shape[0] != 2:
            raise ValueError(f"AABB data must have shape (2, D), got {data.shape}")

        # Validate dimensionality
        dims = data.shape[1]
        if dims not in [2, 3]:
            raise ValueError(f"AABB must be 2D or 3D, got {dims} dimensions")

        # Validate dtype
        if data.dtype not in [np.float32, np.float64]:
            # Try to convert
            data = data.astype(np.float32)

        # Ensure C-contiguous
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)

        # Validate min <= max
        if np.any(data[0] > data[1]):
            raise ValueError("AABB min must be <= max in all dimensions")

        self._data = data
        self._dims = dims
        self._dtype = data.dtype

    @property
    def bounds(self) -> np.ndarray:
        """Get bounds as (2, D) array with [min, max]."""
        return self._data

    @property
    def min(self) -> np.ndarray:
        """Get minimum corner."""
        return self._data[0]

    @property
    def max(self) -> np.ndarray:
        """Get maximum corner."""
        return self._data[1]

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
        return f"AABB(min={self.min.tolist()}, max={self.max.tolist()}, dtype={self._dtype})"
