"""
Dispatch tables for ray casting against spatial structures

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/xlabmedical/trueform
"""

# Dispatch table for spatial ray_cast organized by object type
# Maps spatial_object_type -> function_name_template
# Template expects suffix:
#   - PointCloud: "float2d" or "double3d"
#   - Mesh: "intfloat32d" or "int64double43d"
#   - EdgeMesh: "intfloat2d" or "int64double3d"

_SPATIAL_RAY_CAST_DISPATCH = {
    'PointCloud': "ray_cast_point_cloud_{}",
    'Mesh': "ray_cast_mesh_{}",
    'EdgeMesh': "ray_cast_edge_mesh_{}"
}
