/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./label_connected_components.hpp"
#include "./make_applier.hpp"
#include "./policy/manifold_edge_link.hpp"

namespace tf {
template <typename Index, typename Policy>
auto make_manifold_edge_connected_component_labels(
    const tf::polygons<Policy> &polygons) {
  tf::connected_component_labels<Index> out;
  out.labels.allocate(polygons.size());
  if constexpr (tf::has_manifold_edge_link_policy<Policy>)
    out.n_components = tf::label_connected_components<Index>(
        out.labels, tf::make_applier(polygons.manifold_edge_link()));
  else {
    tf::face_membership<Index> fm;
    fm.build(polygons);
    tf::manifold_edge_link<Index,
                           tf::static_size_v<decltype(polygons.faces()[0])>>
        mel;
    mel.build(polygons.faces(), fm);
    out.n_components = tf::label_connected_components<Index>(
        out.labels, tf::make_applier(mel));
  }
  return out;
}
} // namespace tf
