/*
 * Copyright (c) 2026 XLAB
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
#pragma once
#include "../core/algorithm/parallel_transform.hpp"
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/buffer.hpp"
#include "./arrangement_graph.hpp"
#include "./construct/extract_intersection_curves.hpp"
#include "./impl/loop_connectivity.hpp"
#include <array>
#include <type_traits>

namespace tf {

namespace cut {

/// Core of the arrangement-curve read: the caller supplies the
/// COLLAPSED region connectivity (the csg graph forwards the one its
/// classification already built; the graph-only overload below builds
/// a local one). The seam rules live in
/// @ref tf::cut::extract_intersection_curves; the coplanar-stack mask
/// comes from the graph's pairs.
template <typename RealOut, typename Policy, typename Int, typename Conn>
auto make_intersection_curves(const tf::arrangement_graph<Policy, Int> &graph,
                              const Conn &conn) {
  using Index = typename tf::arrangement_graph<Policy, Int>::index_type;
  const auto &fr = graph.face_regions();
  tf::buffer<char> stacked;
  auto pairs = graph.coplanar_pairs();
  if (pairs.size() != 0) {
    stacked.allocate(fr.loops().size());
    tf::parallel_fill(stacked, char(0));
    for (const auto &cp : pairs) {
      stacked[std::size_t(cp[0])] = char(1);
      stacked[std::size_t(cp[1])] = char(1);
    }
  }
  return extract_intersection_curves<RealOut, Index>(
      fr, graph.triangulations(), conn, tf::make_range(stacked),
      tf::make_range(graph.created_points()), graph.converter());
}

} // namespace cut

/// @ingroup cut
/// @brief The intersection-curve network of an arrangement.
///
/// A seam is a region walk edge with a cross-tag neighbour; within a
/// single tag (a self arrangement) it is a non-manifold edge; and the
/// contact border of a coincident (coplanar) overlap is a seam in both
/// cases, while the overlap's interior stays silent. Under
/// @ref tf::intersect_mode::sos the curves come straight from the
/// intersection graph's edge groups instead.
///
/// @tparam RealOut  Output coordinate type (defaulted to the input's).
/// @return A @ref tf::curves_buffer of the connected polylines.
template <typename RealOut = tf::none_t, typename Policy, typename Int>
auto make_intersection_curves(const tf::arrangement_graph<Policy, Int> &graph) {
  using graph_t = tf::arrangement_graph<Policy, Int>;
  using Index = typename graph_t::index_type;
  using Real =
      std::conditional_t<std::is_same_v<RealOut, tf::none_t>,
                         typename graph_t::input_real_type, RealOut>;

  if (graph.config().intersect.mode & tf::intersect_mode::sos) {
    const auto &ig = graph.intersection_graph();
    tf::buffer<std::array<Index, 2>> edges;
    edges.allocate(std::size_t(ig.edge_groups().size()));
    tf::parallel_transform(
        ig.edge_groups(), tf::make_range(edges),
        [](const auto &group) -> std::array<Index, 2> {
          return {group[0].point_0, group[0].point_1};
        });
    return cut::curves_from_seam_edges<Real, Index>(edges, ig.points(),
                                                    graph.converter());
  }

  tf::buffer<Index> point_counts;
  point_counts.allocate(std::size_t(graph.n_tags()));
  auto apply_to_form = graph.apply_to_form();
  for (Index t = 0; t < graph.n_tags(); ++t)
    apply_to_form(t, [&](const auto &form) {
      point_counts[std::size_t(t)] = static_cast<Index>(form.points().size());
    });
  tf::cut::loop_connectivity<Index> conn;
  conn.build(graph.face_regions(), graph.dead_loops(),
             tf::make_range(point_counts),
             static_cast<Index>(graph.created_points().size()));
  return cut::make_intersection_curves<Real>(
      graph, conn.connectivity_per_carrier_edge());
}

} // namespace tf
