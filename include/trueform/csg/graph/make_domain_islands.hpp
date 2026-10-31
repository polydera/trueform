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
#include "../../core/algorithm/make_equivalence_class_map.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./arrangement_descriptor.hpp"
#include "./compute_keyed_tags.hpp"
#include "tbb/parallel_sort.h"
#include <array>
#include <cstddef>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief The islands of the domain graph, the carriers the seeding cast
///        runs over.
///
/// A component is a wall between the two domains on its sides, and an island
/// is a set of domains its walls join. Two components share a domain only
/// where a fan paired them, which joined their bundle too, so an island lies
/// in one bundle. A bundle whose rings were all certified is one island; a
/// withheld ring states that its components touch, not which of their sides
/// face each other, so its bundle can fall apart into several.
///
/// When every bundle is one island the bundles are the islands, and nothing
/// is materialized: `island_of_component`, `bundle_of_island` and
/// `island_to_tags` stay empty and the descriptor's bundle tables answer.
/// Otherwise bundle `b`'s island is the one holding its smallest component,
/// and every further island is numbered past the bundles in the order of
/// their smallest components.
template <typename Index> struct domain_islands {
  tf::buffer<Index> island_of_component;
  tf::buffer<Index> bundle_of_island;
  tf::offset_block_buffer<Index, Index> island_to_tags;
  Index n_islands = 0;
};

/// @ingroup csg_graph_internals
/// @brief Build the @ref tf::csg::graph::domain_islands of a descriptor.
template <typename Index>
auto make_domain_islands(
    const tf::csg::graph::arrangement_descriptor<Index> &desc)
    -> domain_islands<Index> {
  domain_islands<Index> out;
  const Index n_components = Index(desc.bundle_of_component.size());
  if (desc.n_domains == Index(0) || n_components == Index(0))
    return out;

  tf::buffer<Index> island_of_domain;
  island_of_domain.allocate(static_cast<std::size_t>(desc.n_domains));
  const auto walls = tf::make_mapped_range(
      tf::make_sequence_range(n_components), [&desc](Index c) {
        return std::array<Index, 2>{desc.domain_of_side[2 * c + 0],
                                    desc.domain_of_side[2 * c + 1]};
      });
  out.n_islands =
      Index(tf::make_dense_equivalence_class_map(walls, island_of_domain));
  if (out.n_islands == desc.n_bundles)
    return out;

  const auto n_islands = static_cast<std::size_t>(out.n_islands);
  const auto n_bundles = desc.n_bundles;
  tf::buffer<std::array<Index, 2>> walls_by_island;
  walls_by_island.allocate(static_cast<std::size_t>(n_components));
  tf::parallel_for_each(
      tf::make_sequence_range(n_components),
      [&](Index c) {
        walls_by_island[std::size_t(c)] = {
            island_of_domain[desc.domain_of_side[2 * c]], c};
      },
      tf::checked);
  tbb::parallel_sort(walls_by_island.begin(), walls_by_island.end());
  // per island `{bundle, smallest component, island}`, every island holding
  // a wall of its own domains
  tf::buffer<std::array<Index, 3>> firsts;
  firsts.allocate(n_islands);
  tf::parallel_for_each(
      tf::make_sequence_range(walls_by_island.size()),
      [&](std::size_t at) {
        const auto island = walls_by_island[at][0];
        if (at != 0 && walls_by_island[at - 1][0] == island)
          return;
        const auto c = walls_by_island[at][1];
        firsts[std::size_t(island)] = {desc.bundle_of_component[c], c, island};
      },
      tf::checked);
  tbb::parallel_sort(firsts.begin(), firsts.end());
  // every bundle holds a component, so position `at` in bundle `b`'s run
  // stands behind `b + 1` bundle firsts and `at - b - 1` further islands
  tf::buffer<Index> renamed;
  renamed.allocate(n_islands);
  out.bundle_of_island.allocate(n_islands);
  tf::buffer<std::array<Index, 2>> further;
  further.allocate(n_islands - static_cast<std::size_t>(n_bundles));
  tf::parallel_for_each(
      tf::make_sequence_range(n_islands),
      [&](std::size_t at) {
        const auto b = firsts[at][0];
        if (at == 0 || firsts[at - 1][0] != b) {
          renamed[std::size_t(firsts[at][2])] = b;
          out.bundle_of_island[std::size_t(b)] = b;
        } else
          further[at - std::size_t(b) - 1] = {firsts[at][1], firsts[at][2]};
      },
      tf::checked);
  tbb::parallel_sort(further.begin(), further.end());
  tf::parallel_for_each(
      tf::make_sequence_range(further.size()),
      [&](std::size_t k) {
        const auto island = further[k][1];
        const auto name = n_bundles + Index(k);
        renamed[std::size_t(island)] = name;
        out.bundle_of_island[std::size_t(name)] =
            desc.bundle_of_component[further[k][0]];
      },
      tf::checked);
  out.island_of_component.allocate(static_cast<std::size_t>(n_components));
  tf::parallel_for_each(
      tf::make_sequence_range(walls_by_island.size()),
      [&](std::size_t at) {
        out.island_of_component[std::size_t(walls_by_island[at][1])] =
            renamed[std::size_t(walls_by_island[at][0])];
      },
      tf::checked);
  out.island_to_tags = tf::csg::graph::compute_keyed_tags(
      desc.tag_of_component, out.island_of_component, out.n_islands);
  return out;
}

} // namespace tf::csg::graph
