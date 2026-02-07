"""
Ensure input is a PointCloud object.

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np


def ensure_point_cloud(data, dims: int = None):
    """
    Ensure input is a PointCloud object, wrapping if necessary.

    Parameters
    ----------
    data : PointCloud, ndarray, or tuple
        - PointCloud: returned as-is
        - ndarray: points array wrapped in PointCloud
        - (ndarray, transformation): points + 4x4 transform wrapped in PointCloud
    dims : int, optional
        If provided, validate has this dimensionality (2 or 3).

    Returns
    -------
    PointCloud
        The input as a PointCloud object.

    Raises
    ------
    TypeError
        If data is not a PointCloud, ndarray, or valid tuple.
    ValueError
        If dims is specified and dimensionality doesn't match.

    Examples
    --------
    >>> cloud = ensure_point_cloud(points_array)
    >>> cloud = ensure_point_cloud(existing_cloud)
    >>> cloud = ensure_point_cloud((points, transformation))
    >>> cloud = ensure_point_cloud(data, dims=3)  # Validate 3D
    """
    from .._spatial import PointCloud

    if isinstance(data, PointCloud):
        cloud = data
    elif isinstance(data, np.ndarray):
        cloud = PointCloud(data)
    elif isinstance(data, tuple) and len(data) == 2:
        points, transformation = data
        if not isinstance(points, np.ndarray):
            raise TypeError(
                f"Expected points to be ndarray, got {type(points).__name__}"
            )
        cloud = PointCloud(points)
        cloud.transformation = transformation
    else:
        raise TypeError(
            f"Expected PointCloud, ndarray, or (points, transformation) tuple, "
            f"got {type(data).__name__}"
        )

    if dims is not None and cloud.dims != dims:
        raise ValueError(f"Expected {dims}D, got {cloud.dims}D")

    return cloud
