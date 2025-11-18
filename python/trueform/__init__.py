"""
Trueform - Geometric processing library for Python

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("trueform")
except _metadata.PackageNotFoundError:
    __version__ = "0.0.0"

# Core data structures
from ._spatial import PointCloud, Mesh, EdgeMesh
from ._core import closest_metric_point_pair, closest_metric_point
from ._core import OffsetBlockedArray
# Top-level functions
from .ray_cast import ray_cast
from .intersects import intersects
from .distance import distance, distance2
from .distance_field import distance_field
from ._intersect import isocontours, intersection_curves
from ._cut import isobands, boolean_union, boolean_intersection, boolean_difference
from ._clean import cleaned
from ._reindex import reindex_by_ids, reindex_by_mask, split_into_components, concatenated
from ._topology import label_connected_components
from ._spatial import neighbor_search, gather_intersecting_ids, gather_ids_within_distance
from ._core.transformed import transformed

# IO functions
from ._io import read_stl, write_stl, read_obj, write_obj

# Primitives
from ._primitives import Point, Segment, Polygon, AABB, Ray, Line, Plane

__all__ = [
    # Core
    'PointCloud',
    'Mesh',
    'EdgeMesh',
    'OffsetBlockedArray',
    'closest_metric_point_pair',
    'closest_metric_point',
    'ray_cast',
    'intersects',
    'distance',
    'distance2',
    'distance_field',
    'isocontours',
    'isobands',
    'intersection_curves',
    'boolean_union',
    'boolean_intersection',
    'boolean_difference',
    'cleaned',
    'reindex_by_ids',
    'reindex_by_mask',
    'split_into_components',
    'concatenated',
    'label_connected_components',
    'neighbor_search',
    'gather_intersecting_ids',
    'gather_ids_within_distance',
    'transformed',
    # IO
    'read_stl',
    'write_stl',
    'read_obj',
    'write_obj',
    # Primitives
    'Point', 'Segment', 'Polygon', 'AABB', 'Ray', 'Line', 'Plane',
    '__version__',
]
