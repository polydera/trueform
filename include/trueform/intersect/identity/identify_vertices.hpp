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
#include "../../core/algorithm/make_equivalence_class_map.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../topology/topo_type.hpp"
#include "../classify/intersection_payload.hpp"
#include "../records/tagged_intersection.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::intersect {

/// The canonical flat vertex of an original, read off the identification
/// table. Total: identity is the answer for every vertex no record
/// identified, which is nearly all of them, so the table names only the
/// ones that were.
template <typename Index>
auto canonical_flat_vertex(
    const tf::buffer<std::array<Index, 2>> &identifications, Index flat)
    -> Index {
  const auto it = std::lower_bound(
      identifications.begin(), identifications.end(), flat,
      [](const std::array<Index, 2> &e, Index f) { return e[0] < f; });
  return it != identifications.end() && (*it)[0] == flat ? (*it)[1] : flat;
}

/// Collapse coincident original vertices into one canonical vertex per
/// physical point, over the flat `vertex_offsets` space.
///
/// Every record whose two targets are both vertices witnesses that the
/// two originals are one point — cross-form VV, the coplanar VV of two
/// duplicated vertices in one form, and the shared-vertex records the
/// self duplicator delivers (those name one vertex twice and enter as
/// self-loops). Union-find closes chains no single pair sees; the
/// representative is the lowest flat id, i.e. the lowest (tag, vertex).
///
/// `identifications` is the sparse fact — (flat vertex, canonical flat
/// vertex) ascending in the first — and is empty whenever nothing
/// witnessed a coincidence. @ref canonical_flat_vertex completes it into
/// the total map every later stage keys on.
template <typename Index, typename Int>
auto identify_vertices(
    const tf::buffer<tf::intersect::tagged_intersection<Index>> &records,
    const tf::buffer<tf::exact::edge_fractions<Int, Index>> &parameters,
    tf::buffer<std::array<Index, 2>> &identifications) -> void {
  using intersection_t = tf::intersect::tagged_intersection<Index>;

  tf::buffer<std::array<Index, 2>> pairs;
  tf::generic_generate(tf::make_range(records), pairs,
                       [&](const intersection_t &rec, auto &out) {
                         if (rec.target.label != tf::topo_type::vertex ||
                             rec.target_other.label != tf::topo_type::vertex)
                           return;
                         const auto &ends =
                             parameters[std::size_t(rec.id)];
                         if (ends.from[0] != ends.to[0])
                           out.push_back({ends.from[0], ends.to[0]});
                       },
                       tf::checked(24000));
  if (pairs.size() == 0)
    return;

  tf::buffer<Index> vids;
  vids.allocate(pairs.size() * 2);
  tf::parallel_for_each(
      tf::make_sequence_range(Index(pairs.size())),
      [&](Index i) {
        vids[std::size_t(2 * i)] = pairs[std::size_t(i)][0];
        vids[std::size_t(2 * i + 1)] = pairs[std::size_t(i)][1];
      },
      tf::checked);
  if (vids.size() < 4096)
    std::sort(vids.begin(), vids.end());
  else
    tbb::parallel_sort(vids.begin(), vids.end());
  vids.erase_till_end(std::unique(vids.begin(), vids.end()));

  auto find_idx = [&](Index flat) -> Index {
    return Index(std::lower_bound(vids.begin(), vids.end(), flat) -
                 vids.begin());
  };
  tf::parallel_for_each(
      pairs,
      [&](std::array<Index, 2> &p) { p = {find_idx(p[0]), find_idx(p[1])}; },
      tf::checked);

  tf::buffer<Index> class_map;
  class_map.allocate(vids.size());
  auto n_classes = tf::make_dense_equivalence_class_map(pairs, class_map);

  // `vids` ascends and class ids are minted in scan order, so a class is
  // first seen at its lowest flat id — the class authority.
  tf::buffer<Index> class_rep;
  class_rep.allocate(std::size_t(n_classes));
  Index next = 0;
  for (std::size_t k = 0; k < vids.size(); ++k)
    if (class_map[k] == next)
      class_rep[std::size_t(next++)] = vids[k];

  identifications.allocate(vids.size());
  tf::parallel_for_each(
      tf::make_sequence_range(Index(vids.size())),
      [&](Index k) {
        identifications[std::size_t(k)] = {
            vids[std::size_t(k)],
            class_rep[std::size_t(class_map[std::size_t(k)])]};
      },
      tf::checked);
}

} // namespace tf::intersect
