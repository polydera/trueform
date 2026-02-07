"""
Geometry operations for point cloud alignment

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from .fit_rigid_alignment import fit_rigid_alignment
from .fit_obb_alignment import fit_obb_alignment
from .fit_knn_alignment import fit_knn_alignment
from .fit_icp_alignment import fit_icp_alignment
from .chamfer_error import chamfer_error
from .triangulated import triangulated
from .normals import normals
from .point_normals import point_normals
from .principal_curvatures import principal_curvatures
from .shape_index import shape_index
from .ensure_positive_orientation import ensure_positive_orientation
from .make_mesh_primitives import (
    make_sphere_mesh,
    make_cylinder_mesh,
    make_box_mesh,
    make_plane_mesh,
)
from .measurements import signed_volume, volume, area
from .laplacian_smoothed import laplacian_smoothed
from .taubin_smoothed import taubin_smoothed

__all__ = [
    "fit_rigid_alignment",
    "fit_obb_alignment",
    "fit_knn_alignment",
    "fit_icp_alignment",
    "chamfer_error",
    "triangulated",
    "normals",
    "point_normals",
    "principal_curvatures",
    "shape_index",
    "ensure_positive_orientation",
    "make_sphere_mesh",
    "make_cylinder_mesh",
    "make_box_mesh",
    "make_plane_mesh",
    "signed_volume",
    "volume",
    "area",
    "laplacian_smoothed",
    "taubin_smoothed",
]
