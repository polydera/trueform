/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#include "trueform/python/topology/compute_vertex_link.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace tf::py {

auto register_topology_compute_vertex_link(nanobind::module_ &m) -> void {
  using namespace nanobind;

  // ==========================================================================
  // COMPUTE_VERTEX_LINK - Edge version (Ngon=2)
  // Takes n_ids instead of face_membership
  // Index types: int32, int64
  // Total: 2 bindings
  // ==========================================================================

  // int32, ngon=2 (edges)
  m.def(
      "compute_vertex_link_int_2",
      [](ndarray<numpy, int, shape<-1, 2>> edges, int n_ids) {
        return compute_vertex_link<int>(edges, n_ids);
      },
      arg("edges"), arg("n_ids"));

  // int64, ngon=2 (edges)
  m.def(
      "compute_vertex_link_int64_2",
      [](ndarray<numpy, int64_t, shape<-1, 2>> edges, int64_t n_ids) {
        return compute_vertex_link<int64_t>(edges, n_ids);
      },
      arg("edges"), arg("n_ids"));

  // ==========================================================================
  // COMPUTE_VERTEX_LINK - Face version (Ngon=3,4)
  // Takes face_membership
  // Index types: int32, int64
  // Total: 4 bindings
  // ==========================================================================

  // int32, ngon=3 (triangles)
  m.def(
      "compute_vertex_link_int_3",
      [](ndarray<numpy, int, shape<-1, 3>> faces,
         const offset_blocked_array_wrapper<int, int> &face_membership) {
        return compute_vertex_link<int, 3>(faces, face_membership);
      },
      arg("faces"), arg("face_membership"));

  // int32, ngon=4 (quads)
  m.def(
      "compute_vertex_link_int_4",
      [](ndarray<numpy, int, shape<-1, 4>> faces,
         const offset_blocked_array_wrapper<int, int> &face_membership) {
        return compute_vertex_link<int, 4>(faces, face_membership);
      },
      arg("faces"), arg("face_membership"));

  // int64, ngon=3 (triangles)
  m.def(
      "compute_vertex_link_int64_3",
      [](ndarray<numpy, int64_t, shape<-1, 3>> faces,
         const offset_blocked_array_wrapper<int64_t, int64_t> &face_membership) {
        return compute_vertex_link<int64_t, 3>(faces, face_membership);
      },
      arg("faces"), arg("face_membership"));

  // int64, ngon=4 (quads)
  m.def(
      "compute_vertex_link_int64_4",
      [](ndarray<numpy, int64_t, shape<-1, 4>> faces,
         const offset_blocked_array_wrapper<int64_t, int64_t> &face_membership) {
        return compute_vertex_link<int64_t, 4>(faces, face_membership);
      },
      arg("faces"), arg("face_membership"));
}

} // namespace tf::py
