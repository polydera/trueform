"""
Trueform - Geometric processing library for Python

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

# Core data structures
from .core import PointCloud, closest_metric_point_pair, closest_metric_point

# Top-level functions
from .ray_cast import ray_cast
from .intersects import intersects
from .distance import distance, distance2
from .spatial import neighbor_search

# IO functions
from .io import read_stl

# Primitives
from .primitives import Point, Segment, Polygon, AABB, Ray, Line, Plane

__all__ = [
    # Core
    'PointCloud',
    'closest_metric_point_pair',
    'closest_metric_point',
    'ray_cast',
    'intersects',
    'distance',
    'distance2',
    'neighbor_search',
    # IO
    'read_stl',
    # Primitives
    'Point', 'Segment', 'Polygon', 'AABB', 'Ray', 'Line', 'Plane',
]
