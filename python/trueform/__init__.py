"""
Trueform - Geometric processing library for Python

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

# Core data structures
from ._core import PointCloud, Mesh, EdgeMesh, closest_metric_point_pair, closest_metric_point
from ._core import OffsetBlockedArray
# Top-level functions
from .ray_cast import ray_cast
from .intersects import intersects
from .distance import distance, distance2
from .distance_field import distance_field
from ._intersect import isocontours
from ._cut import isobands
from ._clean import cleaned
from ._spatial import neighbor_search, gather_intersecting_ids, gather_ids_within_distance
from ._core.transformed import transformed

# IO functions
from ._io import read_stl, write_stl

# Primitives
from ._primitives import Point, Segment, Polygon, AABB, Ray, Line, Plane

__all__ = [
    # Core
    'PointCloud',
    'Mesh',
    'EdgeMesh',
    'closest_metric_point_pair',
    'closest_metric_point',
    'ray_cast',
    'intersects',
    'distance',
    'distance2',
    'distance_field',
    'isocontours',
    'isobands',
    'cleaned',
    'neighbor_search',
    'gather_intersecting_ids',
    'gather_ids_within_distance',
    'transformed',
    # IO
    'read_stl',
    'write_stl',
    # Primitives
    'Point', 'Segment', 'Polygon', 'AABB', 'Ray', 'Line', 'Plane',
]
