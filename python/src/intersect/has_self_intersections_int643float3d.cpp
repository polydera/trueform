/*
* Copyright (c) 2025 XLAB
* All rights reserved.
*
* This file is part of trueform (trueform.polydera.com)
*
* Licensed for noncommercial use under the PolyForm Noncommercial
* License 1.0.0.
* Commercial licensing available via info@polydera.com.
*
* Author: Žiga Sajovic
*/

#include "trueform/python/intersect/has_self_intersections.hpp"

namespace tf::py {

auto register_has_self_intersections_int643float3d(nanobind::module_ &m) -> void {
  m.def("has_self_intersections_int643float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh) {
          return has_self_intersections(mesh);
        },
        nanobind::arg("mesh"));
}

} // namespace tf::py
