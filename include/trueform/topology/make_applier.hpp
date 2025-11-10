/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./face_link.hpp"
#include "./manifold_edge_link_like.hpp"
#include "./vertex_link.hpp"

namespace tf {
template <typename Index> auto make_applier(const tf::face_link<Index> &link) {
  return [&link](Index id, const auto &f) {
    for (Index n_id : link[id])
      f(n_id);
  };
}
template <typename Index>
auto make_applier(const tf::vertex_link<Index> &link) {
  return [&link](Index id, const auto &f) {
    for (Index n_id : link[id])
      f(n_id);
  };
}

template <typename Policy>
auto make_applier(const tf::manifold_edge_link_like<Policy> &link) {
  return [link = tf::make_range(link)](auto id, const auto &f) {
    for (const auto &he : link[id])
      if (he.is_simple())
        f(he.face_peer);
  };
}
} // namespace tf
