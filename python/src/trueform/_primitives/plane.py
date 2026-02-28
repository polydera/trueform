"""
Plane primitive

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from .primitive import Primitive, PrimitiveType


class Plane(Primitive):
    """
    A plane defined by a normal vector and offset: dot(normal, x) + offset = 0.

    The data is stored as a flat array of (D+1) elements: [normal..., offset].
    For 2D planes shape is (3,) or (N, 3), for 3D planes shape is (4,) or (N, 4).

    Parameters
    ----------
    data : array-like, optional
        Plane coefficients. Shape (D+1,) for a single plane, (N, D+1) for a
        batch, where D is 2 or 3.
    normal : array-like, optional
        Normal vector(s). Shape (D,) for single, (N, D) for batch.
    offset : float or array-like, optional
        Offset scalar(s). Scalar for single, shape (N,) for batch.

    Either ``data`` or both ``normal`` and ``offset`` must be provided.

    Examples
    --------
    >>> from trueform import Plane
    >>> plane = Plane(normal=[0, 0, 1], offset=-5.0)
    >>> plane.normal
    array([0., 0., 1.], dtype=float32)
    >>> plane.offset
    -5.0
    >>> plane = Plane([0, 0, 1, -5])
    >>> plane.normal
    array([0., 0., 1.], dtype=float32)

    Batch construction:

    >>> import numpy as np
    >>> planes = Plane(np.random.rand(50, 4))
    >>> planes.count
    50
    """

    def __init__(self, data=None, *, normal=None, offset=None, origin=None):
        if data is None:
            normal = np.asarray(normal)
            if origin is not None:
                origin = np.asarray(origin)
                offset = -np.dot(normal, origin)
            offset = np.asarray(offset)
            if normal.ndim == 1:
                data = np.append(normal, offset)
            else:
                data = np.concatenate(
                    [normal, offset[..., np.newaxis]], axis=-1)
        else:
            data = np.asarray(data)
        if data.dtype not in (np.float32, np.float64):
            data = data.astype(np.float32)
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        dims = data.shape[-1] - 1
        if dims != 3:
            raise ValueError(f"Plane requires 3D, got {dims} dimensions")
        self._init_wrapper(data, PrimitiveType.plane, dims)

    @property
    def normal(self) -> np.ndarray:
        """Get normal vector(s)."""
        return self._data[..., :-1]

    @property
    def offset(self):
        """Get offset. Returns scalar for single, array for batch."""
        if self.is_batch:
            return self._data[..., -1]
        return float(self._data[-1])

    @property
    def coeffs(self) -> np.ndarray:
        """Get full coefficient array [normal..., offset]."""
        return self._data

    @classmethod
    def from_point_normal(cls, origin, normal):
        """
        Create plane from a point and normal vector.

        Computes offset as -dot(normal, origin).

        Parameters
        ----------
        origin : array-like
            Point on the plane, shape (D,)
        normal : array-like
            Normal vector, shape (D,)

        Returns
        -------
        Plane

        Examples
        --------
        >>> plane = Plane.from_point_normal([0, 0, 5], [0, 0, 1])
        >>> plane.offset
        -5.0
        """
        origin = np.asarray(origin)
        normal = np.asarray(normal)
        offset = -np.dot(normal, origin)
        return cls(normal=normal, offset=offset)

    @classmethod
    def from_points(cls, p1, p2, p3):
        """
        Create plane passing through three points (3D only).

        Parameters
        ----------
        p1, p2, p3 : array-like
            Points on the plane, each shape (3,)

        Returns
        -------
        Plane

        Examples
        --------
        >>> plane = Plane.from_points([0, 0, 0], [1, 0, 0], [0, 1, 0])
        >>> plane.normal
        array([0., 0., 1.], dtype=float32)
        """
        p1, p2, p3 = np.asarray(p1), np.asarray(p2), np.asarray(p3)
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        offset = -np.dot(normal, p1)
        return cls(normal=normal, offset=offset)

    def __repr__(self) -> str:
        if self.is_batch:
            return f"Plane(batch={self.count}, dims={self._dims}, dtype={self._dtype})"
        return f"Plane(normal={self.normal.tolist()}, offset={self.offset}, dtype={self._dtype})"
