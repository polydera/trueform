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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./arrangement_descriptor.hpp"
#include "./component_crossings.hpp"
#include "./domain_depths.hpp"
#include "./domain_inclusions.hpp"
#include <cstddef>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Multi-source BFS that carries every volume form's depth across the
///        implicit domain graph from the pre-seeded domains, then publishes
///        `depth != 0` as the inclusion bit.
///
/// A domain `d_other` first reached from a visited domain `d` across
/// component `c` inherits `d`'s depth row and then takes `c`'s crossing:
/// every volume layer of the wall adds its signed count, negated when the
/// wall is crossed `side 1 -> side 0`. The layers are
/// @ref tf::csg::graph::component_crossings; a component carrying no
/// coincident stack is its own tag at unit. A sheet bounds no volume, so its
/// layers count nothing and its columns are
/// @ref tf::csg::graph::propagate_sheet_sides's. The seeds are kept as-is —
/// see @ref tf::csg::graph::seed_inclusion_bits for where they come from.
///
/// @param inc               Per-domain inclusion bitvectors; the volume
///                          columns are the product.
/// @param depths            Per-domain depth rows; pre-seeded at every domain
///                          id in `seeds` and dead once this publishes them.
/// @param desc              Arrangement descriptor (`domain_of_side`,
///                          `n_domains`, `tag_of_component`).
/// @param crossings         Per-component layers of the wall it carries.
/// @param domain_components Per domain, the components it bounds.
/// @param is_sheet_tag      Per-form sheet mask (may be empty: no sheets).
/// @param seeds             Pre-seeded domain ids — each becomes a BFS root.
template <typename Index>
auto propagate_inclusion_bits(
    tf::csg::graph::domain_inclusions &inc,
    tf::csg::graph::domain_depths<Index> &depths,
    const tf::csg::graph::arrangement_descriptor<Index> &desc,
    const tf::csg::graph::component_crossings<Index> &crossings,
    const tf::offset_block_buffer<Index, Index> &domain_components,
    const tf::buffer<char> &is_sheet_tag, const tf::buffer<Index> &seeds)
    -> void {
  const Index n_domains = desc.n_domains;
  const std::size_t columns_per_domain = depths.columns_per_domain;

  if (desc.tag_of_component.size() == 0 || n_domains == 0 ||
      seeds.size() == 0)
    return;

  auto domain_of_side_buf = desc.domain_of_side;
  auto tag_of_component = desc.tag_of_component;

  auto sheet_tag = [&](Index t) -> bool {
    return t < Index(is_sheet_tag.size()) && is_sheet_tag[t];
  };

  // `dir` is the direction the wall is crossed in: `+1` entering side 1,
  // `-1` leaving it.
  auto take_crossing = [&](Index d, Index c, Index dir) {
    auto take_layer = [&](Index t, Index delta) {
      if (!sheet_tag(t))
        depths.at(static_cast<std::size_t>(d), static_cast<std::size_t>(t)) +=
            dir * delta;
    };
    const Index row = crossings.row_of(c);
    if (row == Index(-1)) {
      const Index t = tag_of_component[c];
      if (t != Index(-1))
        take_layer(t, Index(1));
      return;
    }
    for (const auto &layer : crossings.layers[static_cast<std::size_t>(row)])
      take_layer(layer[0], layer[1]);
  };

  // Multi-source BFS from `seeds`.
  tf::buffer<char> visited;
  visited.allocate(static_cast<std::size_t>(n_domains));
  tf::parallel_fill(visited, char(0));

  tf::buffer<Index> q;
  std::size_t head = 0;
  for (auto s : seeds) {
    if (s < Index(0) || s >= n_domains)
      continue;
    if (visited[s])
      continue;
    visited[s] = char(1);
    q.push_back(s);
  }

  while (head < q.size()) {
    const Index d = q[head++];
    for (const Index c : domain_components[static_cast<std::size_t>(d)]) {
      const Index d0 = domain_of_side_buf[2 * c + 0];
      const Index d1 = domain_of_side_buf[2 * c + 1];
      const bool from_side0 = d == d0;
      const Index d_other = from_side0 ? d1 : d0;
      if (visited[d_other])
        continue;
      for (std::size_t x = 0; x < columns_per_domain; ++x)
        depths.at(static_cast<std::size_t>(d_other), x) =
            depths.at(static_cast<std::size_t>(d), x);
      take_crossing(d_other, c, from_side0 ? Index(1) : Index(-1));
      visited[d_other] = char(1);
      q.push_back(d_other);
    }
  }

  // A depth is the whole answer for a volume form, so its bit is written
  // both ways: a domain the flood left at zero reads outside.
  tf::parallel_for_each(
      tf::make_sequence_range(n_domains),
      [&](Index d) {
        for (std::size_t t = 0; t < columns_per_domain; ++t) {
          if (sheet_tag(Index(t)))
            continue;
          if (depths.at(static_cast<std::size_t>(d), t) != Index(0))
            inc.set(static_cast<std::size_t>(d), t);
          else
            inc.clear(static_cast<std::size_t>(d), t);
        }
      },
      tf::checked);
}

} // namespace tf::csg::graph
