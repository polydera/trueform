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

#include "trueform/python/topology/compute_manifold_edge_link.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace tf::py {

auto register_topology_compute_manifold_edge_link(nanobind::module_ &m) -> void {
  using namespace nanobind;

  // ==========================================================================
  // COMPUTE_MANIFOLD_EDGE_LINK
  // Index types: int32, int64
  // Ngon: 3, 4
  // Total: 4 bindings
  // ==========================================================================

  // int32, ngon=3 (triangles)
  m.def(
      "compute_manifold_edge_link_int_3",
      [](ndarray<numpy, int, shape<-1, 3>> faces,
         const offset_blocked_array_wrapper<int, int> &face_membership) {
        return compute_manifold_edge_link<int, 3>(faces, face_membership);
      },
      arg("faces"), arg("face_membership"));

  // int32, dynamic
  m.def(
      "compute_manifold_edge_link_int_dyn",
      [](const offset_blocked_array_wrapper<int, int> &faces,
         const offset_blocked_array_wrapper<int, int> &face_membership) {
        return compute_manifold_edge_link_dynamic<int>(faces, face_membership);
      },
      arg("faces"), arg("face_membership"));

  // int64, ngon=3 (triangles)
  m.def(
      "compute_manifold_edge_link_int64_3",
      [](ndarray<numpy, int64_t, shape<-1, 3>> faces,
         const offset_blocked_array_wrapper<int64_t, int64_t> &face_membership) {
        return compute_manifold_edge_link<int64_t, 3>(faces, face_membership);
      },
      arg("faces"), arg("face_membership"));

  // int64, dynamic
  m.def(
      "compute_manifold_edge_link_int64_dyn",
      [](const offset_blocked_array_wrapper<int64_t, int64_t> &faces,
         const offset_blocked_array_wrapper<int64_t, int64_t> &face_membership) {
        return compute_manifold_edge_link_dynamic<int64_t>(faces, face_membership);
      },
      arg("faces"), arg("face_membership"));
}

} // namespace tf::py
