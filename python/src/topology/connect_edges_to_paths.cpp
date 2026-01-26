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

#include "trueform/python/topology/connect_edges_to_paths.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace tf::py {

auto register_topology_connect_edges_to_paths(nanobind::module_ &m) -> void {
  using namespace nanobind;

  // ==========================================================================
  // CONNECT_EDGES_TO_PATHS
  // Index types: int32, int64
  // Total: 2 bindings
  // ==========================================================================

  // int32
  m.def(
      "connect_edges_to_paths_int",
      [](ndarray<numpy, int, shape<-1, 2>> edges) {
        return connect_edges_to_paths<int>(edges);
      },
      arg("edges"));

  // int64
  m.def(
      "connect_edges_to_paths_int64",
      [](ndarray<numpy, int64_t, shape<-1, 2>> edges) {
        return connect_edges_to_paths<int64_t>(edges);
      },
      arg("edges"));
}

} // namespace tf::py
