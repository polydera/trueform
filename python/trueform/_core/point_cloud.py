"""
Core data structures for trueform

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

import numpy as np
from typing import Union
from .._trueform import (
    PointCloudWrapperFloat2D,
    PointCloudWrapperFloat3D,
    PointCloudWrapperDouble2D,
    PointCloudWrapperDouble3D,
)


class PointCloud:
    """
    Point cloud with spatial indexing support.

    Wraps a NumPy array of points and provides efficient spatial queries
    through an internal KD-tree structure.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, D) where N is number of points and D is dimensionality (2 or 3).
        Supports float32 and float64 dtypes.

    Examples
    --------
    >>> import numpy as np
    >>> import trueform as tf
    >>> # 3D points with float32
    >>> points_3d = np.random.rand(1000, 3).astype(np.float32)
    >>> cloud_3d = tf.PointCloud(points_3d)
    >>> cloud_3d.size
    1000
    >>> cloud_3d.dims
    3
    >>> # 2D points with float64
    >>> points_2d = np.random.rand(500, 2).astype(np.float64)
    >>> cloud_2d = tf.PointCloud(points_2d)
    >>> cloud_2d.dims
    2
    """

    def __init__(self, points: np.ndarray, transformation: np.ndarray = None):
        """
        Create a point cloud from a NumPy array.

        Parameters
        ----------
        points : np.ndarray
            Array of shape (N, D) where D is 2 or 3, with dtype float32 or float64
        transformation : np.ndarray, optional
            Transformation matrix (3x3 for 2D, 4x4 for 3D). If provided, applies
            transformation to points during spatial queries.
        """
        # Validate input
        if not isinstance(points, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(points)}")

        if points.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {points.shape}")

        # Check dtype
        if points.dtype not in [np.float32, np.float64]:
            # Try to convert to float32
            points = points.astype(np.float32)

        # Check dimensionality
        dims = points.shape[1]
        if dims not in [2, 3]:
            raise ValueError(f"Expected 2D or 3D points, got {dims} dimensions")

        # Ensure C-contiguous layout for zero-copy views
        if not points.flags['C_CONTIGUOUS']:
            points = np.ascontiguousarray(points)

        # Store the points array (Python owns this data)
        self._points = points

        # Pick the right wrapper based on dtype and dims
        if points.dtype == np.float32:
            if dims == 2:
                self._wrapper = PointCloudWrapperFloat2D(points)
            else:  # dims == 3
                self._wrapper = PointCloudWrapperFloat3D(points)
        else:  # dtype == np.float64
            if dims == 2:
                self._wrapper = PointCloudWrapperDouble2D(points)
            else:  # dims == 3
                self._wrapper = PointCloudWrapperDouble3D(points)

        # Set transformation if provided
        if transformation is not None:
            self.transformation = transformation

    @property
    def points(self) -> np.ndarray:
        """Get the underlying points array."""
        return self._points

    @property
    def size(self) -> int:
        """Get number of points in the cloud."""
        return self._wrapper.size()

    @property
    def dims(self) -> int:
        """Get dimensionality of points."""
        return self._wrapper.dims()

    @property
    def dtype(self) -> np.dtype:
        """Get data type of points (float32 or float64)."""
        return self._points.dtype

    @property
    def transformation(self):
        """
        Get the transformation matrix.

        Returns
        -------
        np.ndarray or None
            Transformation matrix (3x3 for 2D, 4x4 for 3D), or None if not set
        """
        return self._wrapper.transformation()

    @transformation.setter
    def transformation(self, mat: np.ndarray) -> None:
        """
        Set the transformation matrix.

        Parameters
        ----------
        mat : np.ndarray or None
            Transformation matrix (3x3 for 2D points, 4x4 for 3D points).
            Set to None to clear the transformation.
        """
        if mat is None:
            self._wrapper.clear_transformation()
            return

        # Validate matrix shape
        expected_size = self.dims + 1
        if mat.shape != (expected_size, expected_size):
            raise ValueError(
                f"Transformation must be {expected_size}x{expected_size} for {self.dims}D points, "
                f"got shape {mat.shape}"
            )

        # Validate dtype matches points
        if mat.dtype != self._points.dtype:
            raise TypeError(
                f"Transformation dtype ({mat.dtype}) must match points dtype ({self._points.dtype})"
            )

        # Ensure C-contiguous
        if not mat.flags['C_CONTIGUOUS']:
            mat = np.ascontiguousarray(mat)

        self._wrapper.set_transformation(mat)

    def build_tree(self) -> None:
        """
        Build the spatial index tree.

        Call this after modifying the points array to update the spatial index.
        """
        self._wrapper.rebuild_tree()

    def __repr__(self) -> str:
        """String representation of the point cloud."""
        return f"PointCloud({self.size} points, {self.dims}D, dtype={self.dtype})"
