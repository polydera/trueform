/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./path_connector.hpp"

namespace tf {
template <typename Policy>
auto connect_edges_to_paths(const tf::edges<Policy> &edges) {
  using Index = std::decay_t<decltype(edges[0][0])>;
  tf::path_connector<Index, Index> pc;
  pc.build(edges);
  return pc.paths_buffer();
}
} // namespace tf
