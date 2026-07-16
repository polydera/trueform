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
#include "../core/algorithm/generic_generate.hpp"
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/buffer.hpp"
#include "../core/curves_buffer.hpp"
#include "../core/edges.hpp"
#include "../core/views/enumerate.hpp"
#include "../core/views/points.hpp"
#include "../core/views/mapped_range.hpp"
#include "../core/views/zip.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./csg_graph.hpp"
#include <algorithm>
#include <array>
#include <type_traits>

namespace tf {

/// @ingroup csg
/// @brief The intersection-curve network of an N-form CSG arrangement.
///
/// An intersection edge is a region-loop edge that has a neighbour of a
/// different form tag — i.e. where two surfaces cross. Both endpoints are
/// created intersection vertices, whose `vertex_t.id` indexes straight into
/// `intersection_graph().points()`. The edges are connected into polylines
/// via @ref tf::connect_edges_to_paths.
///
/// Coplanar (coincident) walls are *not* picked up: the dead duplicate
/// loop was already removed from the connectivity, so a coincidence leaves
/// no cross-tag neighbour. This mirrors `cut_graph::intersection_edges`,
/// the path `make_mesh_arrangements(.., tf::return_curves)` uses, but reads
/// straight off the `csg_graph` (no separate cut_graph build).
///
/// @return A @ref tf::curves_buffer of the intersection polylines.
template <typename Forms, typename Structs, typename Int>
auto make_intersection_curves(const tf::csg_graph<Forms, Structs, Int> &graph) {
  using graph_t = tf::csg_graph<Forms, Structs, Int>;
  using Index = typename graph_t::index_type;
  using RealOut = typename graph_t::input_real_type;

  const auto &fc = graph.face_cuts();
  const auto &ag = graph.arrangement();
  auto ipts = graph.intersection_graph().points();

  auto loops = fc.loops();
  auto descs = fc.descriptors();
  auto conn = ag.connectivity_per_face_edge();

  // One cross-tag edge per intersection seam, keyed on the created-vertex
  // ids (= intersection_graph().points() indices). Each edge is emitted only
  // from its smallest incident loop id, so the result is already
  // duplicate-free — no sort/unique needed.
  tf::buffer<std::array<Index, 2>> ie;
  tf::generic_generate(
      tf::enumerate(tf::zip(loops, conn)), ie,
      [&](const auto &item, auto &out) {
        auto &&[li_z, tup] = item;
        auto &&[loop, edge_neighbors] = tup;
        // Dead (coplanar-duplicate) loops keep their vertex count but have
        // empty cleaned connectivity — skip them.
        if (edge_neighbors.size() == 0)
          return;
        const Index li = static_cast<Index>(li_z);
        const Index my_tag = descs[li].tag;
        const Index n = static_cast<Index>(loop.size());
        for (Index ei = 0; ei < n; ++ei) {
          bool cross = false, is_min = true;
          for (auto nj : edge_neighbors[ei]) {
            if (descs[nj].tag != my_tag)
              cross = true;
            if (nj < li)
              is_min = false; // a smaller loop on this edge will emit it
          }
          if (cross && is_min) {
            // Cross-tag seam endpoints are always created intersection
            // vertices, so `.id` indexes straight into ig points (see above).
            const Index a = loop[ei].id;
            const Index b = loop[(ei + 1) % n].id;
            out.push_back({std::min(a, b), std::max(a, b)});
          }
        }
      });

  tf::curves_buffer<Index, RealOut, 3> cb;
  cb.paths_buffer() =
      tf::connect_edges_to_paths(tf::make_edges(tf::make_range(ie)));
  cb.points_buffer().allocate(ipts.size());
  if constexpr (std::is_integral_v<RealOut>) {
    tf::parallel_copy(tf::make_points(ipts), cb.points());
  } else {
    auto &conv = graph.converter();
    tf::parallel_copy(tf::make_points(tf::make_mapped_range(
                          ipts, [&conv](const auto &pt) {
                            return conv.deconvert(pt);
                          })),
                      cb.points());
  }
  return cb;
}

} // namespace tf
