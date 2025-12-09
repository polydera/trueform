/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include <nanobind/nanobind.h>
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/topology/orient_faces_consistently.hpp>

namespace tf::py {

auto register_orient_faces_consistently_double2d(nanobind::module_ &m) -> void {

  // ==== double, 2D ====

  // int32, ngon=3
  m.def("orient_faces_consistently_intdouble32d",
        [](mesh_wrapper<int, double, 3, 2> &mesh) {
          return orient_faces_consistently(mesh);
        },
        nanobind::arg("mesh"));

  // int32, dynamic
  m.def("orient_faces_consistently_intdoubledyn2d",
        [](mesh_wrapper<int, double, dynamic_size, 2> &mesh) {
          return orient_faces_consistently(mesh);
        },
        nanobind::arg("mesh"));

  // int64, ngon=3
  m.def("orient_faces_consistently_int64double32d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh) {
          return orient_faces_consistently(mesh);
        },
        nanobind::arg("mesh"));

  // int64, dynamic
  m.def("orient_faces_consistently_int64doubledyn2d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 2> &mesh) {
          return orient_faces_consistently(mesh);
        },
        nanobind::arg("mesh"));
}

} // namespace tf::py
