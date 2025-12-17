/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#include <nanobind/nanobind.h>
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/topology/orient_faces_consistently.hpp>

namespace tf::py {

auto register_orient_faces_consistently_double3d(nanobind::module_ &m) -> void {

  // ==== double, 3D ====

  // int32, ngon=3
  m.def("orient_faces_consistently_intdouble33d",
        [](mesh_wrapper<int, double, 3, 3> &mesh) {
          return orient_faces_consistently(mesh);
        },
        nanobind::arg("mesh"));

  // int32, dynamic
  m.def("orient_faces_consistently_intdoubledyn3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh) {
          return orient_faces_consistently(mesh);
        },
        nanobind::arg("mesh"));

  // int64, ngon=3
  m.def("orient_faces_consistently_int64double33d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh) {
          return orient_faces_consistently(mesh);
        },
        nanobind::arg("mesh"));

  // int64, dynamic
  m.def("orient_faces_consistently_int64doubledyn3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh) {
          return orient_faces_consistently(mesh);
        },
        nanobind::arg("mesh"));
}

} // namespace tf::py
