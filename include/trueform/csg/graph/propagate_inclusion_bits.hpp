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
#include "../../core/views/enumerate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./arrangement_descriptor.hpp"
#include "./component_crossings.hpp"
#include "./domain_depths.hpp"
#include "./domain_inclusions.hpp"
#include <cstddef>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Multi-source BFS that carries every form's depth across the
///        implicit domain graph from the pre-seeded domains, then
///        publishes `depth != 0` as the inclusion bit.
///
/// A domain `d_other` first reached from a visited domain `d` across
/// component `c` inherits both of `d`'s rows — the depths and the packed
/// sheet bits — and then takes `c`'s crossing: every volume layer of the
/// wall adds its signed count, negated when the wall is crossed
/// `side 1 -> side 0`, and every sheet layer flips its bit, a sheet
/// bounding no volume to count. The layers are
/// @ref tf::csg::graph::component_crossings; a component carrying no
/// coincident stack is its own tag at unit. The seeds are kept as-is —
/// see @ref tf::csg::graph::seed_inclusion_bits for where they come from.
///
/// @param inc          Per-domain inclusion bitvectors; pre-seeded at
///                     every domain id in `seeds`, and the product.
/// @param depths       Per-domain depth rows; pre-seeded alongside `inc`
///                     and dead once this publishes them.
/// @param desc         Arrangement descriptor (`domain_of_side`,
///                     `n_domains`, `tag_of_component`).
/// @param n_components The component count the crossings map into.
/// @param crossings    Per-component layers of the wall it carries.
/// @param is_sheet_tag Per-form sheet mask (may be empty: no sheets).
/// @param seeds        Pre-seeded domain ids — each becomes a BFS root.
template <typename Index>
auto propagate_inclusion_bits(
    tf::csg::graph::domain_inclusions &inc,
    tf::csg::graph::domain_depths<Index> &depths,
    const tf::csg::graph::arrangement_descriptor<Index> &desc,
    Index n_components,
    const tf::csg::graph::component_crossings<Index> &crossings,
    const tf::buffer<char> &is_sheet_tag, const tf::buffer<Index> &seeds)
    -> void {
  const Index n_domains = desc.n_domains;
  const std::size_t words_per_domain = inc.words_per_domain;
  const std::size_t columns_per_domain = depths.columns_per_domain;

  if (n_components == 0 || n_domains == 0 || seeds.size() == 0)
    return;

  auto domain_of_side_buf = desc.domain_of_side;
  auto tag_of_component = desc.tag_of_component;

  auto sheet_tag = [&](Index t) -> bool {
    return t < Index(is_sheet_tag.size()) && is_sheet_tag[t];
  };

  // `dir` is the direction the wall is crossed in: `+1` entering side 1,
  // `-1` leaving it.
  auto take_layer = [&](Index d, Index t, Index delta, Index dir) {
    if (sheet_tag(t))
      inc.flip(static_cast<std::size_t>(d), static_cast<std::size_t>(t));
    else
      depths.at(static_cast<std::size_t>(d), static_cast<std::size_t>(t)) +=
          dir * delta;
  };

  auto take_crossing = [&](Index d, Index c, Index dir) {
    const Index row = crossings.row_of(c);
    if (row == Index(-1)) {
      const Index t = tag_of_component[c];
      if (t != Index(-1))
        take_layer(d, t, Index(1), dir);
      return;
    }
    for (const auto &layer : crossings.layers[static_cast<std::size_t>(row)])
      take_layer(d, layer[0], layer[1], dir);
  };

  // Per-domain → incident components via counting sort.
  tf::buffer<Index> in_offsets;
  in_offsets.allocate(static_cast<std::size_t>(n_domains + 1));
  tf::parallel_fill(in_offsets, Index(0));

  for (Index c = 0; c < n_components; ++c) {
    in_offsets[domain_of_side_buf[2 * c + 0] + 1]++;
    in_offsets[domain_of_side_buf[2 * c + 1] + 1]++;
  }
  for (Index i = 0; i < n_domains; ++i)
    in_offsets[i + 1] += in_offsets[i];

  tf::buffer<Index> in_data;
  in_data.allocate(static_cast<std::size_t>(in_offsets[n_domains]));
  {
    tf::buffer<Index> cursor;
    cursor.allocate(static_cast<std::size_t>(n_domains));
    tf::parallel_for_each(
        tf::enumerate(cursor),
        [&in_offsets](auto pair) {
          auto &&[i, c] = pair;
          c = in_offsets[i];
        },
        tf::checked);
    for (Index c = 0; c < n_components; ++c) {
      const Index d0 = domain_of_side_buf[2 * c + 0];
      const Index d1 = domain_of_side_buf[2 * c + 1];
      in_data[cursor[d0]++] = c;
      in_data[cursor[d1]++] = c;
    }
  }

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
    const Index lo = in_offsets[d];
    const Index hi = in_offsets[d + 1];
    for (Index k = lo; k < hi; ++k) {
      const Index c = in_data[k];
      const Index d0 = domain_of_side_buf[2 * c + 0];
      const Index d1 = domain_of_side_buf[2 * c + 1];
      const bool from_side0 = d == d0;
      const Index d_other = from_side0 ? d1 : d0;
      if (visited[d_other])
        continue;
      for (std::size_t w = 0; w < words_per_domain; ++w)
        inc.bits[static_cast<std::size_t>(d_other) * words_per_domain + w] =
            inc.bits[static_cast<std::size_t>(d) * words_per_domain + w];
      for (std::size_t x = 0; x < columns_per_domain; ++x)
        depths.at(static_cast<std::size_t>(d_other), x) =
            depths.at(static_cast<std::size_t>(d), x);
      take_crossing(d_other, c, from_side0 ? Index(1) : Index(-1));
      visited[d_other] = char(1);
      q.push_back(d_other);
    }
  }

  // A depth is the whole answer for a volume form, so its bit is written
  // both ways: a domain the flood left at zero reads outside even where a
  // seed once set it. Sheet columns carry no depth and stay as they are.
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
