/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#include "trueform/python/intersect/self_intersection_curves.hpp"

namespace tf::py {

auto register_self_intersection_curves_int64dyndouble3d(nanobind::module_ &m) -> void {
  // int64, dynamic, float64, 3D
  m.def("self_intersection_curves_mesh_int64dyndouble3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh) {
          return self_intersection_curves(mesh);
        },
        nanobind::arg("mesh"));
}

} // namespace tf::py
