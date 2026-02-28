"""
Dispatch tables for tf::spatial form x form operations.

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""
from .mesh import Mesh
from .edge_mesh import EdgeMesh
from .point_cloud import PointCloud


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
