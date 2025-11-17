"""
Internal dispatch table for form-form gather_ids

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

from .._core.mesh import Mesh
from .._core.edge_mesh import EdgeMesh
from .._core.point_cloud import PointCloud


# Dispatch table for form-form gather_ids
# Maps (form0_type, form1_type) -> (function_name_template, needs_swap)
# needs_swap=True means we need to swap arguments (form1, form0) -> (form0, form1)
_FORM_FORM_GATHER_IDS_DISPATCH = {
    # PointCloud × PointCloud
    (PointCloud, PointCloud): ("gather_ids_point_cloud_point_cloud_{}", False),

    # EdgeMesh × EdgeMesh
    (EdgeMesh, EdgeMesh): ("gather_ids_edge_mesh_edge_mesh_{}", False),

    # EdgeMesh × PointCloud
    (EdgeMesh, PointCloud): ("gather_ids_edge_mesh_point_cloud_{}", False),
    (PointCloud, EdgeMesh): ("gather_ids_edge_mesh_point_cloud_{}", True),

    # Mesh × PointCloud
    (Mesh, PointCloud): ("gather_ids_mesh_point_cloud_{}", False),
    (PointCloud, Mesh): ("gather_ids_mesh_point_cloud_{}", True),

    # Mesh × EdgeMesh
    (Mesh, EdgeMesh): ("gather_ids_mesh_edge_mesh_{}", False),
    (EdgeMesh, Mesh): ("gather_ids_mesh_edge_mesh_{}", True),

    # Mesh × Mesh
    (Mesh, Mesh): ("gather_ids_mesh_mesh_{}", False),
}
