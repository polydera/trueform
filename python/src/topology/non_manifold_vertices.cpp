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

#include "trueform/python/topology/non_manifold_vertices.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace tf::py {

auto register_topology_non_manifold_vertices(nanobind::module_ &m) -> void {
  using namespace nanobind;

  // int32, ngon=3
  m.def(
      "non_manifold_vertices_int_3",
      [](ndarray<numpy, int, shape<-1, 3>> cells,
         const offset_blocked_array_wrapper<int, int> &fm) {
        return non_manifold_vertices<int, 3>(cells, fm);
      },
      arg("cells"), arg("face_membership"));

  // int32, dynamic
  m.def(
      "non_manifold_vertices_int_dyn",
      [](const offset_blocked_array_wrapper<int, int> &cells,
         const offset_blocked_array_wrapper<int, int> &fm) {
        return non_manifold_vertices_dynamic<int>(cells, fm);
      },
      arg("cells"), arg("face_membership"));

  // int64, ngon=3
  m.def(
      "non_manifold_vertices_int64_3",
      [](ndarray<numpy, int64_t, shape<-1, 3>> cells,
         const offset_blocked_array_wrapper<int64_t, int64_t> &fm) {
        return non_manifold_vertices<int64_t, 3>(cells, fm);
      },
      arg("cells"), arg("face_membership"));

  // int64, dynamic
  m.def(
      "non_manifold_vertices_int64_dyn",
      [](const offset_blocked_array_wrapper<int64_t, int64_t> &cells,
         const offset_blocked_array_wrapper<int64_t, int64_t> &fm) {
        return non_manifold_vertices_dynamic<int64_t>(cells, fm);
      },
      arg("cells"), arg("face_membership"));
}

} // namespace tf::py
