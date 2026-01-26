"""
Trueform - Geometric processing library for Python

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"

# Core data structures
from ._spatial import PointCloud, Mesh, EdgeMesh
from ._core import closest_metric_point_pair, closest_metric_point
from ._core import OffsetBlockedArray, as_offset_blocked
# Top-level functions
from .ray_cast import ray_cast
from .intersects import intersects
from .distance import distance, distance2
from .distance_field import distance_field
from ._intersect import isocontours, intersection_curves, self_intersection_curves
from ._cut import isobands, boolean_union, boolean_intersection, boolean_difference, embedded_self_intersection_curves
from ._clean import cleaned
from ._reindex import reindex_by_ids, reindex_by_mask, reindex_by_ids_on_points, reindex_by_mask_on_points, split_into_components, concatenated
from ._topology import label_connected_components, cell_membership, manifold_edge_link, face_link, vertex_link_edges, vertex_link_faces, k_rings, neighborhoods, boundary_edges, boundary_paths, boundary_curves, non_manifold_edges, orient_faces_consistently, connect_edges_to_paths
from ._spatial import neighbor_search, gather_intersecting_ids, gather_ids_within_distance
from ._core.transformed import transformed
from ._geometry import fit_rigid_alignment, fit_obb_alignment, fit_knn_alignment, chamfer_error, triangulated
from ._geometry import normals, point_normals, principal_curvatures, shape_index, ensure_positive_orientation
from ._geometry import make_sphere_mesh, make_cylinder_mesh, make_box_mesh, make_plane_mesh
from ._geometry import signed_volume, volume, area

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
    'as_offset_blocked',
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
    'self_intersection_curves',
    'boolean_union',
    'boolean_intersection',
    'boolean_difference',
    'embedded_self_intersection_curves',
    'cleaned',
    'reindex_by_ids',
    'reindex_by_mask',
    'reindex_by_ids_on_points',
    'reindex_by_mask_on_points',
    'split_into_components',
    'concatenated',
    'label_connected_components',
    'cell_membership',
    'manifold_edge_link',
    'face_link',
    'vertex_link_edges',
    'vertex_link_faces',
    'k_rings',
    'neighborhoods',
    'boundary_edges',
    'boundary_paths',
    'boundary_curves',
    'non_manifold_edges',
    'orient_faces_consistently',
    'connect_edges_to_paths',
    'neighbor_search',
    'gather_intersecting_ids',
    'gather_ids_within_distance',
    'transformed',
    # Geometry
    'fit_rigid_alignment',
    'fit_obb_alignment',
    'fit_knn_alignment',
    'chamfer_error',
    'triangulated',
    'normals',
    'point_normals',
    'principal_curvatures',
    'shape_index',
    'ensure_positive_orientation',
    'make_sphere_mesh',
    'make_cylinder_mesh',
    'make_box_mesh',
    'make_plane_mesh',
    'signed_volume',
    'volume',
    'area',
    # IO
    'read_stl',
    'write_stl',
    'read_obj',
    'write_obj',
    # Primitives
    'Point', 'Segment', 'Polygon', 'AABB', 'Ray', 'Line', 'Plane',
    '__version__',
]
