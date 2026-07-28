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

#include "../../core/buffer.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>

namespace tf::cut {

/// @ingroup cut
/// @brief Collapse merge pairs into one fully compressed class map.
///
/// A merge is an equivalence: the vertex keyed `from` is the vertex keyed
/// `to_key`, whose identity is `to`. The consumer
/// (@ref tf::cut::resolve_region_vertex_key) does a SINGLE binary search,
/// so every `from` must already point at its class representative — no
/// chains, no duplicates, sorted by `from`.
///
/// Collapsed with union-find rather than by iterating a rewrite to a
/// fixpoint. Merge pairs form an undirected equivalence relation and may
/// contain cycles; a directed rewrite has no fixpoint on a cycle of three
/// or more and spins forever. This terminates by construction.
template <typename Index, typename Merge>
auto consolidate_region_merges(tf::buffer<Merge> &merges,
                               tf::buffer<Merge> &incoming) -> bool {
  if (incoming.size() == 0)
    return false;
  const std::size_t before = merges.size();
  for (const auto &merge : incoming)
    merges.push_back(merge);
  incoming.clear();

  using key_t = std::array<Index, 2>;
  tf::buffer<key_t> keys;
  keys.reserve(2 * merges.size());
  for (const auto &merge : merges) {
    keys.push_back(merge.from);
    keys.push_back(merge.to_key);
  }
  std::sort(keys.begin(), keys.end());
  keys.erase_till_end(std::unique(keys.begin(), keys.end()));
  const auto n = static_cast<Index>(keys.size());

  auto id_of = [&](const key_t &key) -> Index {
    return static_cast<Index>(
        std::lower_bound(keys.begin(), keys.end(), key) - keys.begin());
  };

  tf::buffer<Index> parent;
  parent.allocate(std::size_t(n));
  for (Index i = 0; i < n; ++i)
    parent[std::size_t(i)] = i;
  auto find = [&](Index i) {
    while (parent[std::size_t(i)] != i) {
      parent[std::size_t(i)] = parent[std::size_t(parent[std::size_t(i)])];
      i = parent[std::size_t(i)];
    }
    return i;
  };
  for (const auto &merge : merges) {
    const Index a = find(id_of(merge.from));
    const Index b = find(id_of(merge.to_key));
    if (a != b)
      parent[std::size_t(a > b ? a : b)] = a < b ? a : b;
  }

  // Only a key that appeared as a `to_key` carries an identity, and every
  // class holds at least one because every merge contributes one. The
  // representative is the smallest such key in the class, so the choice is
  // independent of the order the pairs arrived in.
  tf::buffer<typename std::decay<decltype(merges[0].to)>::type> vertex;
  tf::buffer<char> has_vertex;
  vertex.allocate(std::size_t(n));
  has_vertex.allocate(std::size_t(n));
  for (Index i = 0; i < n; ++i)
    has_vertex[std::size_t(i)] = 0;
  for (const auto &merge : merges) {
    const Index i = id_of(merge.to_key);
    vertex[std::size_t(i)] = merge.to;
    has_vertex[std::size_t(i)] = 1;
  }

  tf::buffer<Index> representative;
  representative.allocate(std::size_t(n));
  for (Index i = 0; i < n; ++i)
    representative[std::size_t(i)] = Index(-1);
  for (Index i = 0; i < n; ++i) {
    if (!has_vertex[std::size_t(i)])
      continue;
    const Index root = find(i);
    if (representative[std::size_t(root)] == Index(-1))
      representative[std::size_t(root)] = i;
  }

  merges.clear();
  for (Index i = 0; i < n; ++i) {
    const Index rep = representative[std::size_t(find(i))];
    if (rep == Index(-1) || rep == i)
      continue;
    merges.push_back({keys[std::size_t(i)], keys[std::size_t(rep)],
                      vertex[std::size_t(rep)]});
  }
  return merges.size() != before;
}

} // namespace tf::cut
