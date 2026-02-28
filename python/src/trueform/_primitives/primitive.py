"""
Primitive base class and PrimitiveType enum

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import enum
import numpy as np
from .. import _trueform


class PrimitiveType(enum.IntEnum):
    point = 0
    segment = 1
    triangle = 2
    ray = 3
    line = 4
    plane = 5
    aabb = 6
    polygon = 7


# Map (dtype, dims) -> C++ wrapper class
_WRAPPER_MAP = {
    (np.dtype('float32'), 2): _trueform.core.PrimitiveWrapperFloat2D,
    (np.dtype('float32'), 3): _trueform.core.PrimitiveWrapperFloat3D,
    (np.dtype('float64'), 2): _trueform.core.PrimitiveWrapperDouble2D,
    (np.dtype('float64'), 3): _trueform.core.PrimitiveWrapperDouble3D,
}


def _make_wrapper(data, prim_type, dims):
    """Create the correct C++ primitive_wrapper variant."""
    key = (data.dtype, dims)
    cls = _WRAPPER_MAP.get(key)
    if cls is None:
        raise ValueError(f"Unsupported dtype/dims combination: {dtype}, {dims}D")
    return cls(int(prim_type), data)


class Primitive:
    """
    Base class for all geometric primitives.

    Wraps a C++ primitive_wrapper that holds the type enum and numpy data together,
    enabling efficient dispatch in C++ without per-type Python→C++ boundary crossings.

    Subclasses: Point, Segment, Triangle, Ray, Line, Plane, AABB, Polygon
    """

    __slots__ = ('_wrapper', '_data', '_dims', '_dtype')

    def _init_wrapper(self, data: np.ndarray, prim_type: PrimitiveType, dims: int):
        """Initialize the C++ wrapper from validated data."""
        self._data = data
        self._dims = dims
        self._dtype = data.dtype
        self._wrapper = _make_wrapper(data, prim_type, dims)

    @property
    def type(self) -> PrimitiveType:
        """Get the primitive type enum."""
        return PrimitiveType(self._wrapper.type_int())

    @property
    def data(self) -> np.ndarray:
        """Get underlying numpy data array."""
        return self._data

    @property
    def array(self) -> np.ndarray:
        """Alias for data."""
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
    def is_batch(self) -> bool:
        """Whether this holds a batch of primitives."""
        return self._wrapper.is_batch()

    @property
    def count(self) -> int:
        """Number of primitives (1 for single, N for batch)."""
        return self._wrapper.count()

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, key):
        """
        NumPy-style indexing on the batch dimension.

        Returns the same primitive type. Integer indexing returns a single
        primitive; slice/array indexing returns a batch.

        Only supported on batches.
        """
        if not self.is_batch:
            raise IndexError("Cannot index a single primitive; indexing is only supported on batches")
        return type(self)(self._data[key])
