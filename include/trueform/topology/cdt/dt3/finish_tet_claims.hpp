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
#include "../../../core/algorithm/parallel_contains.hpp"
#include "../../../core/algorithm/reduce.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/config/verify.hpp"
#include "../../../core/views/mapped_range.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "../../tetrahedralization_refusal.hpp"
#include "./clear_tetrahedralization_cells.hpp"
#include "./compact_tet_dead_rows.hpp"
#include "./insert_tetrahedralization_site.hpp"
#include "./release_tet_claim_pools.hpp"
#include "./tet_claim_word.hpp"
#include "./tet_claim_workspace.hpp"
#include "./tet_insertion.hpp"
#include "./tet_site_position.hpp"
#include "./tet_state.hpp"
#include <atomic>
#include <cstddef>
#include <functional>

namespace tf::topology::cdt::dt3 {

/// Close the streams' insertions: their killed edges and dead rows join the
/// owner's, the pools give their rows back, the hint stands on a live row, the
/// serial tail inserts what the streams left, and the dead rows compact at the
/// owner's own threshold. The rows are left in no canonical order. The bands
/// inserted at least one site, so a live row stands for the hint.
template <typename Owner, typename Index>
auto finish_tet_claims(Owner &owner, tet_claim_workspace<Index> &work) -> bool {
  for (auto &killed : work.killed) {
    for (const auto &edge : killed)
      owner._killed.push_back(edge);
    killed.clear();
  }
  long long dead = 0;
  for (const auto &stream : work.streams)
    dead += stream.dead_delta;
  owner._n_dead = std::size_t(static_cast<long long>(owner._n_dead) + dead);
  release_tet_claim_pools(owner, work);
  if constexpr (tf::core::verification)
    tf::core::verify(!tf::parallel_contains(
                         tf::make_sequence_range(work.ownership.size()),
                         [&work](std::size_t i) {
                           return work.ownership[i].load(
                                      std::memory_order_relaxed) != tet_unowned;
                         },
                         tf::checked),
                     "the insertions end with every word released");
  if constexpr (tf::core::verification)
    tf::core::verify(
        tf::reduce(tf::make_mapped_range(owner._states,
                                         [](tet_state state) {
                                           return std::size_t(state ==
                                                              tet_state::dead);
                                         }),
                   std::plus<>{}, std::size_t(0), tf::checked) == owner._n_dead,
        "the dead count is the dead rows'");
  if constexpr (tf::core::verification)
    tf::core::verify(work.inserted + work.tail.size() == tet_site_count(owner),
                     "the bands insert every site but the tail's");
  owner._canonical = false;
  const std::size_t rows = std::size_t(owner._corners.size());
  std::size_t hint = 0;
  while (hint < rows && owner._states[hint] == tet_state::dead)
    ++hint;
  if constexpr (tf::core::verification)
    tf::core::verify(hint < rows,
                     "the bands leave a live row for the hint to stand on");
  owner._hint = Index(hint < rows ? hint : 0);
  for (auto site : work.tail)
    if (insert_tetrahedralization_site(owner, site) != tet_insertion::placed) {
      clear_tetrahedralization_cells(owner);
      owner._refusal = tf::tetrahedralization_refusal::index_capacity;
      return false;
    }
  compact_tet_dead_rows<true>(owner);
  return true;
}

} // namespace tf::topology::cdt::dt3
