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

#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/views/sequence_range.hpp"

#include "tbb/parallel_sort.h"

#include <array>
#include <cstddef>
#include <limits>

namespace tf::csg::graph {

/// The rank of every component class in an order no schedule can move: a
/// class is keyed by its first member — its smallest exposed-triangle row,
/// or, for a class of uncut faces alone, the first such face after every row,
/// tag-major — and the classes are ranked by key.
///
/// `class_of(node)` is a node's class; `triangle_nodes` and `polygon_nodes`
/// name each exposed triangle's and each uncut face's node, `none` where
/// nothing is named. A class's first member always starts a run of equal
/// nodes, so only run starts are gathered.
template <typename Index, typename ClassOf, typename TriangleNodes,
          typename PolygonNodes>
auto make_canonical_component_ranks(const ClassOf &class_of,
                                    const TriangleNodes &triangle_nodes,
                                    const PolygonNodes &polygon_nodes,
                                    Index n_classes, Index none)
    -> tf::buffer<Index> {
  using key_t = std::array<Index, 2>;
  const auto n_rows = Index(triangle_nodes.size());
  tf::buffer<key_t> firsts;
  tf::generic_generate(
      tf::make_sequence_range(n_rows), firsts,
      [&](Index e, tf::buffer<key_t> &out) {
        const auto node = triangle_nodes[std::size_t(e)];
        if (node == none ||
            (e != 0 && triangle_nodes[std::size_t(e) - 1] == node))
          return;
        out.push_back({class_of(node), e});
      },
      tf::checked);
  Index base = n_rows;
  for (const auto &nodes : polygon_nodes) {
    const auto n_faces = Index(nodes.size());
    tf::buffer<key_t> tag_firsts;
    tf::generic_generate(
        tf::make_sequence_range(n_faces), tag_firsts,
        [&](Index f, tf::buffer<key_t> &out) {
          const auto node = nodes[std::size_t(f)];
          if (node == none || (f != 0 && nodes[std::size_t(f) - 1] == node))
            return;
          out.push_back({class_of(node), base + f});
        },
        tf::checked);
    tf::core::append(tag_firsts, firsts);
    base += n_faces;
  }
  tbb::parallel_sort(firsts.begin(), firsts.end());

  // each class's key is its first record; a class nothing names keeps the
  // largest key and ranks after every named one, by its own id
  tf::buffer<key_t> keys;
  keys.allocate(std::size_t(n_classes));
  tf::parallel_for_each(
      tf::make_sequence_range(n_classes),
      [&](Index c) {
        keys[std::size_t(c)] = {std::numeric_limits<Index>::max(), c};
      },
      tf::checked);
  for (std::size_t at = 0; at < firsts.size(); ++at)
    if (at == 0 || firsts[at - 1][0] != firsts[at][0])
      keys[std::size_t(firsts[at][0])][0] = firsts[at][1];
  tbb::parallel_sort(keys.begin(), keys.end());

  tf::buffer<Index> rank;
  rank.allocate(std::size_t(n_classes));
  tf::parallel_for_each(
      tf::make_sequence_range(n_classes),
      [&](Index r) { rank[std::size_t(keys[std::size_t(r)][1])] = r; },
      tf::checked);
  return rank;
}

} // namespace tf::csg::graph
