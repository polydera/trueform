"""
Internal dispatch table for intersects with spatial forms (Mesh, EdgeMesh, PointCloud)

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/xlabmedical/trueform
"""

from .mesh import Mesh
from .edge_mesh import EdgeMesh
from .point_cloud import PointCloud
from .._primitives import Point, Segment, Polygon, Line, Ray, Plane


# Dispatch table for spatial intersects (form-primitive)
# Maps (form_type, primitive_type) -> (function_name_template, needs_swap)
# needs_swap=True means we need to swap arguments (primitive, form) -> (form, primitive)
_INTERSECTS_DISPATCH = {
    # Mesh combinations
    (Mesh, Point): ("intersects_mesh_point_{}", False),
    (Point, Mesh): ("intersects_mesh_point_{}", True),
    (Mesh, Segment): ("intersects_mesh_segment_{}", False),
    (Segment, Mesh): ("intersects_mesh_segment_{}", True),
    (Mesh, Polygon): ("intersects_mesh_polygon_{}", False),
    (Polygon, Mesh): ("intersects_mesh_polygon_{}", True),
    (Mesh, Ray): ("intersects_mesh_ray_{}", False),
    (Ray, Mesh): ("intersects_mesh_ray_{}", True),
    (Mesh, Line): ("intersects_mesh_line_{}", False),
    (Line, Mesh): ("intersects_mesh_line_{}", True),
    (Mesh, Plane): ("intersects_mesh_plane_{}", False),
    (Plane, Mesh): ("intersects_mesh_plane_{}", True),

    # EdgeMesh combinations
    (EdgeMesh, Point): ("intersects_edge_mesh_point_{}", False),
    (Point, EdgeMesh): ("intersects_edge_mesh_point_{}", True),
    (EdgeMesh, Segment): ("intersects_edge_mesh_segment_{}", False),
    (Segment, EdgeMesh): ("intersects_edge_mesh_segment_{}", True),
    (EdgeMesh, Polygon): ("intersects_edge_mesh_polygon_{}", False),
    (Polygon, EdgeMesh): ("intersects_edge_mesh_polygon_{}", True),
    (EdgeMesh, Ray): ("intersects_edge_mesh_ray_{}", False),
    (Ray, EdgeMesh): ("intersects_edge_mesh_ray_{}", True),
    (EdgeMesh, Line): ("intersects_edge_mesh_line_{}", False),
    (Line, EdgeMesh): ("intersects_edge_mesh_line_{}", True),
    (EdgeMesh, Plane): ("intersects_edge_mesh_plane_{}", False),
    (Plane, EdgeMesh): ("intersects_edge_mesh_plane_{}", True),

    # PointCloud combinations
    (PointCloud, Point): ("intersects_point_cloud_point_{}", False),
    (Point, PointCloud): ("intersects_point_cloud_point_{}", True),
    (PointCloud, Segment): ("intersects_point_cloud_segment_{}", False),
    (Segment, PointCloud): ("intersects_point_cloud_segment_{}", True),
    (PointCloud, Polygon): ("intersects_point_cloud_polygon_{}", False),
    (Polygon, PointCloud): ("intersects_point_cloud_polygon_{}", True),
    (PointCloud, Ray): ("intersects_point_cloud_ray_{}", False),
    (Ray, PointCloud): ("intersects_point_cloud_ray_{}", True),
    (PointCloud, Line): ("intersects_point_cloud_line_{}", False),
    (Line, PointCloud): ("intersects_point_cloud_line_{}", True),
    (PointCloud, Plane): ("intersects_point_cloud_plane_{}", False),
    (Plane, PointCloud): ("intersects_point_cloud_plane_{}", True),

    # Form-Form combinations
    (PointCloud, PointCloud): ("intersects_point_cloud_point_cloud_{}", False),
    (EdgeMesh, EdgeMesh): ("intersects_edge_mesh_edge_mesh_{}", False),
    (EdgeMesh, PointCloud): ("intersects_edge_mesh_point_cloud_{}", False),
    (PointCloud, EdgeMesh): ("intersects_edge_mesh_point_cloud_{}", True),
    (Mesh, PointCloud): ("intersects_mesh_point_cloud_{}", False),
    (PointCloud, Mesh): ("intersects_mesh_point_cloud_{}", True),
    (Mesh, EdgeMesh): ("intersects_mesh_edge_mesh_{}", False),
    (EdgeMesh, Mesh): ("intersects_mesh_edge_mesh_{}", True),
    (Mesh, Mesh): ("intersects_mesh_mesh_{}", False),
}
