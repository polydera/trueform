"""
Internal dispatch table for form-form neighbor_search

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

from .mesh import Mesh
from .edge_mesh import EdgeMesh
from .point_cloud import PointCloud


# Dispatch table for form-form neighbor_search
# Maps (form0_type, form1_type) -> (function_name_template, needs_swap)
# needs_swap=True means we need to swap arguments (form1, form0) -> (form0, form1)
_FORM_FORM_NEIGHBOR_SEARCH_DISPATCH = {
    # PointCloud × PointCloud
    (PointCloud, PointCloud): ("neighbor_search_point_cloud_point_cloud_{}", False),

    # EdgeMesh × EdgeMesh
    (EdgeMesh, EdgeMesh): ("neighbor_search_edge_mesh_edge_mesh_{}", False),

    # EdgeMesh × PointCloud
    (EdgeMesh, PointCloud): ("neighbor_search_edge_mesh_point_cloud_{}", False),
    (PointCloud, EdgeMesh): ("neighbor_search_edge_mesh_point_cloud_{}", True),

    # Mesh × PointCloud
    (Mesh, PointCloud): ("neighbor_search_mesh_point_cloud_{}", False),
    (PointCloud, Mesh): ("neighbor_search_mesh_point_cloud_{}", True),

    # Mesh × EdgeMesh
    (Mesh, EdgeMesh): ("neighbor_search_mesh_edge_mesh_{}", False),
    (EdgeMesh, Mesh): ("neighbor_search_mesh_edge_mesh_{}", True),

    # Mesh × Mesh
    (Mesh, Mesh): ("neighbor_search_mesh_mesh_{}", False),
}
