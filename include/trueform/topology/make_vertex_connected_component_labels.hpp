/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./face_membership.hpp"
#include "./label_connected_components.hpp"
#include "./make_applier.hpp"
#include "./policy/vertex_link.hpp"

namespace tf {
template <typename Policy>
auto make_vertex_connected_component_labels(
    const tf::polygons<Policy> &polygons) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  tf::connected_component_labels<Index> out;
  out.labels.allocate(polygons.points().size());
  if constexpr (tf::has_vertex_link_policy<Policy>)
    out.n_components = tf::label_connected_components<Index>(
        out.labels, tf::make_applier(polygons.vertex_link()));
  else {
    tf::face_membership<Index> fm;
    fm.build(polygons);
    tf::vertex_link<Index> vl;
    vl.build(polygons, fm);
    out.n_components = tf::label_connected_components<Index>(
        out.labels, tf::make_applier(vl));
  }
  return out;
}
} // namespace tf
