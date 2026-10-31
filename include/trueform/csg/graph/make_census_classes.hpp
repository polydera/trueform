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
#include "../../core/buffer.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::csg::graph {

/// The identity classes the pairs state: one block per class of two or more
/// ids joined by a pair, its members ascending, the blocks ordered by their
/// smallest member. A pair is read by its two ids; its witness planes never
/// enter closure; no member is elected — the statement elects.
template <typename Index, std::size_t N>
auto make_census_classes(const tf::buffer<std::array<Index, N>> &pairs)
    -> tf::offset_block_buffer<Index, Index> {
  tf::buffer<Index> ids;
  ids.allocate(pairs.size() * 2);
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    ids[2 * i] = pairs[i][0];
    ids[2 * i + 1] = pairs[i][1];
  }
  tbb::parallel_sort(ids.begin(), ids.end());
  ids.erase_till_end(std::unique(ids.begin(), ids.end()));
  const auto slot_of = [&](Index id) {
    return Index(std::lower_bound(ids.begin(), ids.end(), id) - ids.begin());
  };
  tf::buffer<std::array<Index, 2>> slot_pairs;
  slot_pairs.allocate(pairs.size());
  for (std::size_t i = 0; i < pairs.size(); ++i)
    slot_pairs[i] = {slot_of(pairs[i][0]), slot_of(pairs[i][1])};
  // class ids are dealt in ascending slot order, so a class's id orders it
  // by its smallest member
  tf::buffer<Index> class_of;
  class_of.allocate(ids.size());
  const auto n_classes =
      std::size_t(tf::make_dense_equivalence_class_map(slot_pairs, class_of));
  const auto none = static_cast<Index>(-1);
  tf::buffer<Index> cursor;
  cursor.allocate(n_classes);
  std::fill(cursor.begin(), cursor.end(), Index(0));
  for (const auto c : class_of)
    ++cursor[std::size_t(c)];
  tf::offset_block_buffer<Index, Index> classes;
  auto &offsets = classes.offsets_buffer();
  offsets.reserve(n_classes + 1);
  offsets.push_back(Index(0));
  for (auto &at : cursor) {
    if (at < Index(2)) {
      at = none;
      continue;
    }
    const auto begin = offsets.back();
    offsets.push_back(Index(begin + at));
    at = begin;
  }
  auto &members = classes.data_buffer();
  members.allocate(std::size_t(offsets.back()));
  for (std::size_t slot = 0; slot < ids.size(); ++slot) {
    auto &at = cursor[std::size_t(class_of[slot])];
    if (at != none)
      members[std::size_t(at++)] = ids[slot];
  }
  return classes;
}

} // namespace tf::csg::graph
