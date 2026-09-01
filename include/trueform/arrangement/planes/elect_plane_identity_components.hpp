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
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>

namespace tf::arrangement {

template <typename Index> using plane_identity_key = std::array<Index, 2>;

template <typename Index>
using plane_identity_relation = std::array<plane_identity_key<Index>, 2>;

/// Elect one identity for every sparse equivalence component.
///
/// Key ranks are the priority law: zero is an existing created identity,
/// one is a fresh class, and two is an original vertex. The caller owns only
/// the two mint operations; union and election have one producer.
template <typename Index, typename MintClass, typename MintOriginal>
auto elect_plane_identity_components(
    tf::buffer<plane_identity_key<Index>> &keys,
    const tf::buffer<plane_identity_relation<Index>> &relations,
    const MintClass &mint_class, const MintOriginal &mint_original,
    tf::buffer<Index> &parent, tf::buffer<Index> &target_of_root,
    tf::buffer<std::array<Index, 3>> &merges) -> void {
  keys.reserve(keys.size() + relations.size() * 2);
  for (const auto &relation : relations) {
    keys.push_back(relation[0]);
    keys.push_back(relation[1]);
  }
  tbb::parallel_sort(keys.begin(), keys.end());
  keys.erase_till_end(std::unique(keys.begin(), keys.end()));
  if (keys.size() == 0) {
    parent.clear();
    target_of_root.clear();
    merges.clear();
    return;
  }

  const auto n_keys = Index(keys.size());
  const auto id_of = [&](const plane_identity_key<Index> &key) {
    return Index(std::lower_bound(keys.begin(), keys.end(), key) -
                 keys.begin());
  };
  parent.allocate(std::size_t(n_keys));
  for (Index i = 0; i < n_keys; ++i)
    parent[std::size_t(i)] = i;
  const auto find = [&](Index value) {
    while (parent[std::size_t(value)] != value) {
      parent[std::size_t(value)] =
          parent[std::size_t(parent[std::size_t(value)])];
      value = parent[std::size_t(value)];
    }
    return value;
  };
  for (const auto &relation : relations) {
    const auto a = find(id_of(relation[0]));
    const auto b = find(id_of(relation[1]));
    if (a != b)
      parent[std::size_t(a > b ? a : b)] = a < b ? a : b;
  }
  for (Index i = 0; i < n_keys; ++i)
    parent[std::size_t(i)] = find(i);

  // the ranks lead the key order, so each priority pass owns a contiguous span
  const auto rank_begin = [&keys](Index rank) {
    return Index(
        std::lower_bound(
            keys.begin(), keys.end(),
            plane_identity_key<Index>{rank,
                                      std::numeric_limits<Index>::min()}) -
        keys.begin());
  };
  const auto fresh_at = rank_begin(Index(1));
  const auto original_at = rank_begin(Index(2));
  const auto beyond_at = rank_begin(Index(3));
  target_of_root.allocate(std::size_t(n_keys));
  std::fill(target_of_root.begin(), target_of_root.end(), Index(-1));
  for (Index i = 0; i < fresh_at; ++i) {
    const auto &key = keys[std::size_t(i)];
    auto &target = target_of_root[std::size_t(parent[std::size_t(i)])];
    if (target == Index(-1) || key[1] < target)
      target = key[1];
  }
  for (Index i = fresh_at; i < original_at; ++i) {
    auto &target = target_of_root[std::size_t(parent[std::size_t(i)])];
    if (target == Index(-1))
      target = mint_class(keys[std::size_t(i)][1]);
  }
  for (Index i = original_at; i < beyond_at; ++i) {
    auto &target = target_of_root[std::size_t(parent[std::size_t(i)])];
    if (target == Index(-1))
      target = mint_original(keys[std::size_t(i)][1]);
  }

  merges.clear();
  merges.reserve(std::size_t(fresh_at) + std::size_t(beyond_at - original_at));
  for (Index i = 0; i < fresh_at; ++i) {
    const auto &key = keys[std::size_t(i)];
    const auto target = target_of_root[std::size_t(parent[std::size_t(i)])];
    if (key[1] != target)
      merges.push_back({Index(1), key[1], target});
  }
  for (Index i = original_at; i < beyond_at; ++i)
    merges.push_back({Index(0), keys[std::size_t(i)][1],
                      target_of_root[std::size_t(parent[std::size_t(i)])]});
  tbb::parallel_sort(merges.begin(), merges.end());
  merges.erase_till_end(std::unique(merges.begin(), merges.end()));
}

} // namespace tf::arrangement
