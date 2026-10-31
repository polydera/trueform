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
#include "../../core/algorithm/parallel_transform.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./identity_substrate.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tf::arrangement {

/// What one created identity is in a published product: the three accounts
/// the substrate's verbs leave (@ref identity_substrate), then each way a
/// product can break them.
enum class created_identity_account : std::int8_t {
  live,
  retired,
  vanished,
  unaccounted,
  emitted_dead
};

/// Per created identity `x` in `[0, n_created)`, one rule at both tiers:
/// `retired` when `P∘L` takes it anywhere, `live` when no layer retires it
/// and some corner of `triangles` is `n_flat + x`, and `vanished` when no
/// layer retires it and nothing references it — no corner and none of
/// `named`, the created ids the product's rows name, ascending. A triangle
/// naming a retired identity and an identity a row names but no corner
/// emits are each their own account.
template <typename Index, typename Int, typename Triangles, typename Named>
auto account_created_identities(
    const identity_substrate<Index, Int> &identities,
    const Triangles &triangles, const Named &named, Index n_flat,
    Index n_created) -> tf::buffer<created_identity_account> {
  tf::buffer<Index> emitted;
  tf::generic_generate(triangles, emitted,
                       [n_flat](const auto &triangle, tf::buffer<Index> &out) {
                         for (const auto corner : triangle)
                           if (Index(corner) >= n_flat)
                             out.push_back(Index(corner) - n_flat);
                       });
  tbb::parallel_sort(emitted.begin(), emitted.end());
  emitted.erase_till_end(std::unique(emitted.begin(), emitted.end()));
  const auto holds = [](const auto &ids, Index id) {
    return std::binary_search(ids.begin(), ids.end(), id);
  };

  tf::buffer<created_identity_account> accounts;
  accounts.allocate(std::size_t(n_created));
  tf::parallel_transform(
      tf::make_sequence_range(n_created), accounts,
      [&](Index id) {
        const bool is_emitted = holds(emitted, id);
        if (identities.composed_target(Index(1), id) != Index(-1))
          return is_emitted ? created_identity_account::emitted_dead
                            : created_identity_account::retired;
        if (is_emitted)
          return created_identity_account::live;
        return holds(named, id) ? created_identity_account::unaccounted
                                : created_identity_account::vanished;
      },
      tf::checked);
  return accounts;
}

} // namespace tf::arrangement
