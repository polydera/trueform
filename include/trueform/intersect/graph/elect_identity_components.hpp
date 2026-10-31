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
#include "../../core/config/verify.hpp"
#include "../../core/range.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/slice.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::intersect::graph {

/// A candidate of THE ELECTION, `{rank, id}`. The rank is the priority law —
/// `0` a standing created identity, `1` a fresh statement no identity speaks
/// yet, `2` an original vertex — and it leads the key order, so the
/// smallest key of a coincidence is the one that speaks it.
template <typename Index> using identity_key = std::array<Index, 2>;

inline constexpr int standing_identity_rank = 0;
inline constexpr int fresh_identity_rank = 1;
inline constexpr int original_identity_rank = 2;

template <typename Index>
auto standing_identity_key(Index id) -> identity_key<Index> {
  return {Index(standing_identity_rank), id};
}
template <typename Index>
auto fresh_identity_key(Index statement) -> identity_key<Index> {
  return {Index(fresh_identity_rank), statement};
}
template <typename Index>
auto original_identity_key(Index flat) -> identity_key<Index> {
  return {Index(original_identity_rank), flat};
}

/// Two candidates the caller states are one identity.
template <typename Index>
using identity_relation = std::array<identity_key<Index>, 2>;

/// The election of a coincidence that retires nothing — no original and at
/// most one standing created identity: its smallest key speaks it, and no
/// row is stated.
template <typename Index, typename Component, typename Mint>
auto elect_identity_component(const Component &component, const Mint &mint)
    -> Index {
  if constexpr (tf::core::verification)
    tf::core::verify(
        component.size() != 0 &&
            std::is_sorted(component.begin(), component.end()) &&
            identity_key<Index>(component[component.size() - 1])[0] !=
                Index(original_identity_rank) &&
            (component.size() == 1 || identity_key<Index>(component[1])[0] ==
                                          Index(fresh_identity_rank)),
        "a coincidence that retires nothing holds no original and one "
        "standing identity at most");
  const identity_key<Index> winner = component[0];
  return winner[0] == Index(standing_identity_rank) ? winner[1] : mint(winner);
}

/// THE ELECTION — one coincidence, one identity, the one law every tier's
/// identity statement is made by. `component` holds the candidates one
/// coincidence joins, ascending, and its smallest key speaks it: a standing
/// created identity survives; otherwise the coincidence mints, and
/// `mint(winner)` numbers that identity in the caller's created space, in
/// the order the caller states its coincidences. Every other standing
/// created identity states `{1, id, target}` and every original
/// `{0, id, target}` into `merges` — originals never survive a coincidence —
/// while a fresh statement states no row and speaks the returned target.
/// Which candidates one coincidence joins is the caller's statement.
template <typename Index, typename Component, typename Mint>
auto elect_identity_component(const Component &component, const Mint &mint,
                              tf::buffer<std::array<Index, 3>> &merges)
    -> Index {
  if constexpr (tf::core::verification)
    tf::core::verify(component.size() != 0 &&
                         std::is_sorted(component.begin(), component.end()),
                     "a coincidence states its candidates ascending");
  const identity_key<Index> winner = component[0];
  const auto target =
      winner[0] == Index(standing_identity_rank) ? winner[1] : mint(winner);
  for (const identity_key<Index> key : component) {
    if (key[0] == Index(standing_identity_rank)) {
      if (key[1] != target)
        merges.push_back({Index(1), key[1], target});
    } else if (key[0] == Index(original_identity_rank))
      merges.push_back({Index(0), key[1], target});
  }
  return target;
}

/// The election over sparse relations: `keys` — the candidates no relation
/// names — and the two ends of every relation close under `relations` into
/// coincidences, elected in the order of their smallest keys. `keys` leaves
/// sorted and unique, `targets` holds per key the identity it speaks, and
/// `merges` the rows, sorted and unique.
template <typename Index, typename Mint>
auto elect_identity_components(
    tf::buffer<identity_key<Index>> &keys,
    const tf::buffer<identity_relation<Index>> &relations, const Mint &mint,
    tf::buffer<Index> &targets, tf::buffer<std::array<Index, 3>> &merges)
    -> void {
  merges.clear();
  keys.reserve(keys.size() + relations.size() * 2);
  for (const auto &relation : relations) {
    keys.push_back(relation[0]);
    keys.push_back(relation[1]);
  }
  tbb::parallel_sort(keys.begin(), keys.end());
  keys.erase_till_end(std::unique(keys.begin(), keys.end()));
  const auto n_keys = Index(keys.size());
  targets.allocate(std::size_t(n_keys));
  if (n_keys == Index(0))
    return;

  const auto id_of = [&](const identity_key<Index> &key) {
    return Index(std::lower_bound(keys.begin(), keys.end(), key) -
                 keys.begin());
  };
  // the keys ascend, so the classes are numbered by first appearance in key
  // order: class `c` is the coincidence of the `c`-th smallest key that
  // opens one
  tf::buffer<Index> class_of;
  class_of.allocate(std::size_t(n_keys));
  const auto n_classes = tf::make_dense_equivalence_class_map(
      tf::make_mapped_range(relations,
                            [&](const identity_relation<Index> &relation) {
                              return std::array<Index, 2>{id_of(relation[0]),
                                                          id_of(relation[1])};
                            }),
      class_of);
  tf::buffer<Index> offsets;
  offsets.allocate_and_initialize(std::size_t(n_classes) + 1, Index(0));
  for (const auto cls : class_of)
    ++offsets[std::size_t(cls) + 1];
  for (std::size_t cls = 1; cls < offsets.size(); ++cls)
    offsets[cls] += offsets[cls - 1];
  tf::buffer<Index> members;
  members.allocate(std::size_t(n_keys));
  {
    tf::buffer<Index> cursor;
    cursor.allocate(std::size_t(n_classes));
    std::copy(offsets.begin(), offsets.end() - 1, cursor.begin());
    for (Index i = 0; i < n_keys; ++i)
      members[std::size_t(cursor[std::size_t(class_of[std::size_t(i)])]++)] = i;
  }

  for (Index cls = 0; cls < n_classes; ++cls) {
    const auto coincidence = tf::slice(
        tf::make_range(members), std::size_t(offsets[std::size_t(cls)]),
        std::size_t(offsets[std::size_t(cls) + 1]));
    const auto target = elect_identity_component<Index>(
        tf::make_indirect_range(coincidence, keys), mint, merges);
    for (const auto at : coincidence)
      targets[std::size_t(at)] = target;
  }
  tbb::parallel_sort(merges.begin(), merges.end());
  merges.erase_till_end(std::unique(merges.begin(), merges.end()));
}

} // namespace tf::intersect::graph
