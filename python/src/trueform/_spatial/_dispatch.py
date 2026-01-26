"""
Dispatch tables for tf::spatial operations (forms and primitives).

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""
from .mesh import Mesh
from .edge_mesh import EdgeMesh
from .point_cloud import PointCloud
from .._primitives import Point, Segment, Polygon, Ray, Line, Plane


# =============================================================================
# NEIGHBOR_SEARCH: FormType -> {PrimitiveType: func_template}
# =============================================================================
NEIGHBOR_SEARCH = {
    Mesh: {
        Point: "neighbor_search_mesh_point_{}",
        Segment: "neighbor_search_mesh_segment_{}",
        Polygon: "neighbor_search_mesh_polygon_{}",
        Ray: "neighbor_search_mesh_ray_{}",
        Line: "neighbor_search_mesh_line_{}",
        Plane: "neighbor_search_mesh_plane_{}",
    },
    EdgeMesh: {
        Point: "neighbor_search_edge_mesh_point_{}",
        Segment: "neighbor_search_edge_mesh_segment_{}",
        Polygon: "neighbor_search_edge_mesh_polygon_{}",
        Ray: "neighbor_search_edge_mesh_ray_{}",
        Line: "neighbor_search_edge_mesh_line_{}",
        Plane: "neighbor_search_edge_mesh_plane_{}",
    },
    PointCloud: {
        Point: "neighbor_search_point_{}",
        Segment: "neighbor_search_segment_{}",
        Polygon: "neighbor_search_polygon_{}",
        Ray: "neighbor_search_ray_{}",
        Line: "neighbor_search_line_{}",
        Plane: "neighbor_search_plane_{}",
    },
}


# =============================================================================
# NEIGHBOR_SEARCH_KNN: FormType -> {PrimitiveType: func_template}
# =============================================================================
NEIGHBOR_SEARCH_KNN = {
    Mesh: {
        Point: "neighbor_search_mesh_knn_point_{}",
        Segment: "neighbor_search_mesh_knn_segment_{}",
        Polygon: "neighbor_search_mesh_knn_polygon_{}",
        Ray: "neighbor_search_mesh_knn_ray_{}",
        Line: "neighbor_search_mesh_knn_line_{}",
        Plane: "neighbor_search_mesh_knn_plane_{}",
    },
    EdgeMesh: {
        Point: "neighbor_search_edge_mesh_knn_point_{}",
        Segment: "neighbor_search_edge_mesh_knn_segment_{}",
        Polygon: "neighbor_search_edge_mesh_knn_polygon_{}",
        Ray: "neighbor_search_edge_mesh_knn_ray_{}",
        Line: "neighbor_search_edge_mesh_knn_line_{}",
        Plane: "neighbor_search_edge_mesh_knn_plane_{}",
    },
    PointCloud: {
        Point: "neighbor_search_knn_point_{}",
        Segment: "neighbor_search_knn_segment_{}",
        Polygon: "neighbor_search_knn_polygon_{}",
        Ray: "neighbor_search_knn_ray_{}",
        Line: "neighbor_search_knn_line_{}",
        Plane: "neighbor_search_knn_plane_{}",
    },
}


# =============================================================================
# GATHER_IDS_FORM_PRIM: (FormType, PrimitiveType) -> (func_template, needs_swap)
# =============================================================================
GATHER_IDS_FORM_PRIM = {
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
}


# =============================================================================
# GATHER_IDS_FORM_FORM: (FormType0, FormType1) -> (func_template, needs_swap)
# =============================================================================
GATHER_IDS_FORM_FORM = {
    # PointCloud x PointCloud
    (PointCloud, PointCloud): ("gather_ids_point_cloud_point_cloud_{}", False),

    # EdgeMesh x EdgeMesh
    (EdgeMesh, EdgeMesh): ("gather_ids_edge_mesh_edge_mesh_{}", False),

    # EdgeMesh x PointCloud
    (EdgeMesh, PointCloud): ("gather_ids_edge_mesh_point_cloud_{}", False),
    (PointCloud, EdgeMesh): ("gather_ids_edge_mesh_point_cloud_{}", True),

    # Mesh x PointCloud
    (Mesh, PointCloud): ("gather_ids_mesh_point_cloud_{}", False),
    (PointCloud, Mesh): ("gather_ids_mesh_point_cloud_{}", True),

    # Mesh x EdgeMesh
    (Mesh, EdgeMesh): ("gather_ids_mesh_edge_mesh_{}", False),
    (EdgeMesh, Mesh): ("gather_ids_mesh_edge_mesh_{}", True),

    # Mesh x Mesh
    (Mesh, Mesh): ("gather_ids_mesh_mesh_{}", False),
}


# =============================================================================
# INTERSECTS_FORM_PRIM: (FormType, PrimitiveType) -> (func_template, needs_swap)
# =============================================================================
INTERSECTS_FORM_PRIM = {
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
}


# =============================================================================
# INTERSECTS_FORM_FORM: (FormType0, FormType1) -> (func_template, needs_swap)
# =============================================================================
INTERSECTS_FORM_FORM = {
    # PointCloud x PointCloud
    (PointCloud, PointCloud): ("intersects_point_cloud_point_cloud_{}", False),

    # EdgeMesh x EdgeMesh
    (EdgeMesh, EdgeMesh): ("intersects_edge_mesh_edge_mesh_{}", False),

    # EdgeMesh x PointCloud
    (EdgeMesh, PointCloud): ("intersects_edge_mesh_point_cloud_{}", False),
    (PointCloud, EdgeMesh): ("intersects_edge_mesh_point_cloud_{}", True),

    # Mesh x PointCloud
    (Mesh, PointCloud): ("intersects_mesh_point_cloud_{}", False),
    (PointCloud, Mesh): ("intersects_mesh_point_cloud_{}", True),

    # Mesh x EdgeMesh
    (Mesh, EdgeMesh): ("intersects_mesh_edge_mesh_{}", False),
    (EdgeMesh, Mesh): ("intersects_mesh_edge_mesh_{}", True),

    # Mesh x Mesh
    (Mesh, Mesh): ("intersects_mesh_mesh_{}", False),
}


# =============================================================================
# RAY_CAST: FormType -> func_template
# =============================================================================
RAY_CAST = {
    PointCloud: "ray_cast_point_cloud_{}",
    Mesh: "ray_cast_mesh_{}",
    EdgeMesh: "ray_cast_edge_mesh_{}",
}


# =============================================================================
# NEIGHBOR_SEARCH_FORM_FORM: (FormType0, FormType1) -> (func_template, needs_swap)
# =============================================================================
NEIGHBOR_SEARCH_FORM_FORM = {
    # PointCloud x PointCloud
    (PointCloud, PointCloud): ("neighbor_search_point_cloud_point_cloud_{}", False),

    # EdgeMesh x EdgeMesh
    (EdgeMesh, EdgeMesh): ("neighbor_search_edge_mesh_edge_mesh_{}", False),

    # EdgeMesh x PointCloud
    (EdgeMesh, PointCloud): ("neighbor_search_edge_mesh_point_cloud_{}", False),
    (PointCloud, EdgeMesh): ("neighbor_search_edge_mesh_point_cloud_{}", True),

    # Mesh x PointCloud
    (Mesh, PointCloud): ("neighbor_search_mesh_point_cloud_{}", False),
    (PointCloud, Mesh): ("neighbor_search_mesh_point_cloud_{}", True),

    # Mesh x EdgeMesh
    (Mesh, EdgeMesh): ("neighbor_search_mesh_edge_mesh_{}", False),
    (EdgeMesh, Mesh): ("neighbor_search_mesh_edge_mesh_{}", True),

    # Mesh x Mesh
    (Mesh, Mesh): ("neighbor_search_mesh_mesh_{}", False),
}
