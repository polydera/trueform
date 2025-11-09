"""
Internal dispatch table for ray casting against core primitives

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

from ..primitives import Segment, Polygon, Line, AABB, Plane


# Dispatch table for core ray_cast (ray to primitives)
# Maps target_type -> function_name_template
_RAY_CAST_DISPATCH = {
    Plane: "ray_cast_ray_plane_{}",
    Polygon: "ray_cast_ray_polygon_{}",
    Segment: "ray_cast_ray_segment_{}",
    Line: "ray_cast_ray_line_{}",
    AABB: "ray_cast_ray_aabb_{}",
}
