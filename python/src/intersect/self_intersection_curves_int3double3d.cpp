/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#include "trueform/python/intersect/self_intersection_curves.hpp"

namespace tf::py {

auto register_self_intersection_curves_int3double3d(nanobind::module_ &m) -> void {
  // int32, triangles, float64, 3D
  m.def("self_intersection_curves_mesh_int3double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh) {
          return self_intersection_curves(mesh);
        },
        nanobind::arg("mesh"));
}

} // namespace tf::py
