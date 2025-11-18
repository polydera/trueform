/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/range.hpp"
#include "./face_link_like.hpp"
#include "./manifold_edge_link_like.hpp"
#include "./vertex_link_like.hpp"

namespace tf {
template <typename Policy>
auto make_applier(const tf::face_link_like<Policy> &link) {
  return [link = tf::make_range(link)](auto id, const auto &f) {
    for (auto n_id : link[id])
      f(n_id);
  };
}
template <typename Policy>
auto make_applier(const tf::vertex_link_like<Policy> &link) {
  return [link = tf::make_range(link)](auto id, const auto &f) {
    for (auto n_id : link[id])
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
