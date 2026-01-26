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

#include "trueform/python/topology/compute_cell_membership.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace tf::py {

auto register_topology_compute_cell_membership(nanobind::module_ &m) -> void {
  using namespace nanobind;

  // ==========================================================================
  // COMPUTE_CELL_MEMBERSHIP
  // Index types: int32, int64
  // Ngon: 2, 3, 4
  // Total: 6 bindings
  // ==========================================================================

  // int32, ngon=2 (edges)
  m.def(
      "compute_cell_membership_int_2",
      [](ndarray<numpy, int, shape<-1, 2>> cells, int n_ids) {
        return compute_cell_membership<int, 2>(cells, n_ids);
      },
      arg("cells"), arg("n_ids"));

  // int32, ngon=3 (triangles)
  m.def(
      "compute_cell_membership_int_3",
      [](ndarray<numpy, int, shape<-1, 3>> cells, int n_ids) {
        return compute_cell_membership<int, 3>(cells, n_ids);
      },
      arg("cells"), arg("n_ids"));

  // int32, dynamic
  m.def(
      "compute_cell_membership_int_dyn",
      [](const offset_blocked_array_wrapper<int, int> &cells, int n_ids) {
        return compute_cell_membership_dynamic<int>(cells, n_ids);
      },
      arg("cells"), arg("n_ids"));

  // int64, ngon=2 (edges)
  m.def(
      "compute_cell_membership_int64_2",
      [](ndarray<numpy, int64_t, shape<-1, 2>> cells, int64_t n_ids) {
        return compute_cell_membership<int64_t, 2>(cells, n_ids);
      },
      arg("cells"), arg("n_ids"));

  // int64, ngon=3 (triangles)
  m.def(
      "compute_cell_membership_int64_3",
      [](ndarray<numpy, int64_t, shape<-1, 3>> cells, int64_t n_ids) {
        return compute_cell_membership<int64_t, 3>(cells, n_ids);
      },
      arg("cells"), arg("n_ids"));

  // int64, dynamic
  m.def(
      "compute_cell_membership_int64_dyn",
      [](const offset_blocked_array_wrapper<int64_t, int64_t> &cells,
         int64_t n_ids) {
        return compute_cell_membership_dynamic<int64_t>(cells, n_ids);
      },
      arg("cells"), arg("n_ids"));
}

} // namespace tf::py
