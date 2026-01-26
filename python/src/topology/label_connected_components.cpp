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

#include "trueform/python/topology/label_connected_components_impl.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

namespace tf::py {

auto register_topology_label_connected_components(nanobind::module_ &m)
    -> void {
  using namespace nanobind;

  // ==========================================================================
  // LABEL_CONNECTED_COMPONENTS - OffsetBlockedArray versions
  // Index types: int32, int64
  // Total: 2 bindings
  // ==========================================================================

  // int32 OffsetBlockedArray
  m.def(
      "label_connected_components_offset_blocked_int",
      [](const offset_blocked_array_wrapper<int, int> &conn,
         std::optional<int> expected_number_of_components) {
        return label_connected_components_impl<int>(
            conn, expected_number_of_components);
      },
      arg("conn"), arg("expected_number_of_components").none() = nanobind::none());

  // int64 OffsetBlockedArray
  m.def(
      "label_connected_components_offset_blocked_int64",
      [](const offset_blocked_array_wrapper<int64_t, int64_t> &conn,
         std::optional<int64_t> expected_number_of_components) {
        return label_connected_components_impl<int64_t>(
            conn, expected_number_of_components);
      },
      arg("conn"), arg("expected_number_of_components").none() = nanobind::none());

  // ==========================================================================
  // LABEL_CONNECTED_COMPONENTS - ndarray versions (fixed-width connectivity)
  // Index types: int32, int64
  // Total: 2 bindings
  // ==========================================================================

  // int32 ndarray
  m.def(
      "label_connected_components_ndarray_int",
      [](ndarray<numpy, const int, shape<-1, -1>> conn,
         std::optional<int> expected_number_of_components) {
        return label_connected_components_impl<int>(
            conn, expected_number_of_components);
      },
      arg("conn"), arg("expected_number_of_components").none() = nanobind::none());

  // int64 ndarray
  m.def(
      "label_connected_components_ndarray_int64",
      [](ndarray<numpy, const int64_t, shape<-1, -1>> conn,
         std::optional<int64_t> expected_number_of_components) {
        return label_connected_components_impl<int64_t>(
            conn, expected_number_of_components);
      },
      arg("conn"), arg("expected_number_of_components").none() = nanobind::none());
}

} // namespace tf::py
