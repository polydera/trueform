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

auto register_orient_faces_consistently_float2d(nanobind::module_ &m) -> void {

  // ==== float, 2D ====

  // int32, ngon=3
  m.def("orient_faces_consistently_intfloat32d",
        [](mesh_wrapper<int, float, 3, 2> &mesh) {
          return orient_faces_consistently(mesh);
        },
        nanobind::arg("mesh"));

  // int32, dynamic
  m.def("orient_faces_consistently_intfloatdyn2d",
        [](mesh_wrapper<int, float, dynamic_size, 2> &mesh) {
          return orient_faces_consistently(mesh);
        },
        nanobind::arg("mesh"));

  // int64, ngon=3
  m.def("orient_faces_consistently_int64float32d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh) {
          return orient_faces_consistently(mesh);
        },
        nanobind::arg("mesh"));

  // int64, dynamic
  m.def("orient_faces_consistently_int64floatdyn2d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 2> &mesh) {
          return orient_faces_consistently(mesh);
        },
        nanobind::arg("mesh"));
}

} // namespace tf::py
