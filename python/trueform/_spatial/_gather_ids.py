"""
Internal dispatch table for gather_ids with spatial forms (Mesh, EdgeMesh, PointCloud)

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

from .._core.mesh import Mesh
from .._core.edge_mesh import EdgeMesh
from .._core.point_cloud import PointCloud
from .._primitives import Point, Segment, Polygon, Line, Ray


# Dispatch table for spatial gather_ids (form-primitive only)
# Maps (form_type, primitive_type) -> (function_name_template, needs_swap)
# needs_swap=True means we need to swap arguments (primitive, form) -> (form, primitive)
_GATHER_IDS_DISPATCH = {
    # Mesh combinations
    (Mesh, Point): ("gather_ids_point_{}", False),
    (Point, Mesh): ("gather_ids_point_{}", True),
    (Mesh, Segment): ("gather_ids_segment_{}", False),
    (Segment, Mesh): ("gather_ids_segment_{}", True),
    (Mesh, Polygon): ("gather_ids_polygon_{}", False),
    (Polygon, Mesh): ("gather_ids_polygon_{}", True),
    (Mesh, Ray): ("gather_ids_ray_{}", False),
    (Ray, Mesh): ("gather_ids_ray_{}", True),
    (Mesh, Line): ("gather_ids_line_{}", False),
    (Line, Mesh): ("gather_ids_line_{}", True),

    # EdgeMesh combinations
    (EdgeMesh, Point): ("gather_ids_point_{}", False),
    (Point, EdgeMesh): ("gather_ids_point_{}", True),
    (EdgeMesh, Segment): ("gather_ids_segment_{}", False),
    (Segment, EdgeMesh): ("gather_ids_segment_{}", True),
    (EdgeMesh, Polygon): ("gather_ids_polygon_{}", False),
    (Polygon, EdgeMesh): ("gather_ids_polygon_{}", True),
    (EdgeMesh, Ray): ("gather_ids_ray_{}", False),
    (Ray, EdgeMesh): ("gather_ids_ray_{}", True),
    (EdgeMesh, Line): ("gather_ids_line_{}", False),
    (Line, EdgeMesh): ("gather_ids_line_{}", True),

    # PointCloud combinations
    (PointCloud, Point): ("gather_ids_point_{}", False),
    (Point, PointCloud): ("gather_ids_point_{}", True),
    (PointCloud, Segment): ("gather_ids_segment_{}", False),
    (Segment, PointCloud): ("gather_ids_segment_{}", True),
    (PointCloud, Polygon): ("gather_ids_polygon_{}", False),
    (Polygon, PointCloud): ("gather_ids_polygon_{}", True),
    (PointCloud, Ray): ("gather_ids_ray_{}", False),
    (Ray, PointCloud): ("gather_ids_ray_{}", True),
    (PointCloud, Line): ("gather_ids_line_{}", False),
    (Line, PointCloud): ("gather_ids_line_{}", True),

    # Form-Form combinations
    (PointCloud, PointCloud): ("gather_ids_point_cloud_point_cloud_{}", False),
    (EdgeMesh, EdgeMesh): ("gather_ids_edge_mesh_edge_mesh_{}", False),
    (EdgeMesh, PointCloud): ("gather_ids_edge_mesh_point_cloud_{}", False),
    (PointCloud, EdgeMesh): ("gather_ids_edge_mesh_point_cloud_{}", True),
    (Mesh, EdgeMesh): ("gather_ids_mesh_edge_mesh_{}", False),
    (EdgeMesh, Mesh): ("gather_ids_mesh_edge_mesh_{}", True),
    (Mesh, PointCloud): ("gather_ids_mesh_point_cloud_{}", False),
    (PointCloud, Mesh): ("gather_ids_mesh_point_cloud_{}", True),
    (Mesh, Mesh): ("gather_ids_mesh_mesh_{}", False),
}
