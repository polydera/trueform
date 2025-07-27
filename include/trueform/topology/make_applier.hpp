/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./face_link.hpp"
#include "./manifold_edge_link.hpp"
#include "./vertex_link.hpp"

namespace tf {
template <typename Index> auto make_applier(const tf::face_link<Index>& link) {
  return [&link](Index id, const auto &f) {
    for (Index n_id : link[id])
      f(n_id);
  };
}
template <typename Index> auto make_applier(const tf::vertex_link<Index>& link) {
  return [&link](Index id, const auto &f) {
    for (Index n_id : link[id])
      f(n_id);
  };
}

template <typename Index, std::size_t N>
auto make_applier(const tf::manifold_edge_link<Index, N>& link) {
  return [&link](Index id, const auto &f) {
    for (const tf::manifold_edge_peer<Index> &he : link[id])
      if (he.is_simple())
        f(he.face_peer);
  };
}
} // namespace tf
