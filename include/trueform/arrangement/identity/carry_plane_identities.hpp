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

#include "../../core/algorithm/parallel_contains.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/parallel_transform.hpp"
#include "../../core/checked.hpp"
#include "../../core/config/verify.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./extent_shift.hpp"
#include "./plane_identity_tables.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <utility>

namespace tf::arrangement {

/// The plane layer `prior` stated, carried across `created` into this value's
/// currency. Only `crossing_kind` and `refinement_kind` names speak flat
/// endpoints the shift reaches; the rewrite rows stay closed and ordered, and
/// the founded planes' names cross verbatim (@ref
/// identity_substrate::carry_plane).
template <typename Index, typename Int>
auto carry_plane_identities(plane_identity_tables<Index, Int> &&prior,
                            const extent_shift<Index> &created, Index n_flat,
                            Index crossing_kind, Index refinement_kind,
                            plane_identity_tables<Index, Int> &identities)
    -> void {
  using name_t = typename plane_identity_tables<Index, Int>::name_t;
  identities.created_class = std::move(prior.created_class);
  identities.direct_points = std::move(prior.direct_points);
  identities.class_t = std::move(prior.class_t);
  identities.class_name = std::move(prior.class_name);
  identities.plane_names = std::move(prior.plane_names);
  if (created.base != created.prior) {
    const auto flat = created.lifted(n_flat);
    tf::parallel_for_each(
        identities.class_name,
        [&](name_t &name) {
          if (name.feature[0] == crossing_kind ||
              name.feature[0] == refinement_kind)
            for (std::size_t f = 1; f < 5; ++f)
              name.feature[f] = flat.forward(name.feature[f]);
        },
        tf::checked);
  }
  identities.merges.allocate(prior.merges.size());
  tf::parallel_transform(
      prior.merges, identities.merges,
      [&](std::array<Index, 3> row) {
        if (row[0] == Index(1))
          row[1] = created.forward(row[1]);
        row[2] = created.forward(row[2]);
        return row;
      },
      tf::checked);
  if constexpr (tf::core::verification) {
    const auto &merges = identities.merges;
    tf::core::verify(
        !tf::parallel_contains(
            tf::make_sequence_range(std::size_t(1),
                                    std::max(merges.size(), std::size_t(1))),
            [&merges](std::size_t at) {
              return !(std::tie(merges[at - 1][0], merges[at - 1][1]) <
                       std::tie(merges[at][0], merges[at][1]));
            },
            tf::checked),
        "a forwarded table keeps its key order");
  }
}

} // namespace tf::arrangement
