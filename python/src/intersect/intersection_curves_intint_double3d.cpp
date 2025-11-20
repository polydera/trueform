/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include "trueform/python/intersect/intersection_curves.hpp"

namespace tf::py {

auto register_intersection_curves_intint_double3d(nanobind::module_ &m) -> void {
  // int32 × int32, double, 3D

  m.def("intersection_curves_mesh_mesh_intint33double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int, double, 3, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_intint34double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int, double, 4, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_intint43double3d",
        [](mesh_wrapper<int, double, 4, 3> &mesh0,
           mesh_wrapper<int, double, 3, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_intint44double3d",
        [](mesh_wrapper<int, double, 4, 3> &mesh0,
           mesh_wrapper<int, double, 4, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });
}

} // namespace tf::py
