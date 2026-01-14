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

#include "trueform/python/topology/make_k_rings.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace tf::py {

auto register_topology_make_k_rings(nanobind::module_ &m) -> void {
  using namespace nanobind;

  // ==========================================================================
  // MAKE_K_RINGS
  // Takes connectivity (vertex link) and k (number of rings)
  // Index types: int32, int64
  // Total: 2 bindings
  // ==========================================================================

  // int32
  m.def(
      "make_k_rings_int",
      [](const offset_blocked_array_wrapper<int, int> &connectivity,
         std::size_t k, bool inclusive) {
        return make_k_rings<int>(connectivity, k, inclusive);
      },
      arg("connectivity"), arg("k"), arg("inclusive") = false);

  // int64
  m.def(
      "make_k_rings_int64",
      [](const offset_blocked_array_wrapper<int64_t, int64_t> &connectivity,
         std::size_t k, bool inclusive) {
        return make_k_rings<int64_t>(connectivity, k, inclusive);
      },
      arg("connectivity"), arg("k"), arg("inclusive") = false);
}

} // namespace tf::py
