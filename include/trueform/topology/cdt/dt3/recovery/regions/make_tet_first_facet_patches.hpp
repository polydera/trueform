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
#include "../../../../../core/algorithm/parallel_for_each.hpp"
#include "../../../../../core/buffer.hpp"
#include "../../../../../core/checked.hpp"
#include "../../../../../core/config/verify.hpp"
#include "../../../../../core/views/sequence_range.hpp"
#include "../../tet_claim_word.hpp"
#include "../../tet_claim_words.hpp"
#include "./lower_tet_claim_word.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Per id below `n_ids`, the patch `patch_of_facet` gives the smallest facet
/// naming it, `ids_of(f, g)` calling `g` on every id facet `f` names, into
/// `patch_of_id`; `-1` for an id no facet names. A site's facets are its
/// triangles' and a segment's the rims running along it: every facet lowers
/// the claim word in `claims` of each id it names to its own, in parallel,
/// so the facets' ids are never turned round and the answer is the same at
/// any worker count.
template <typename IdsOf, typename Index>
auto make_tet_first_facet_patches(const IdsOf &ids_of,
                                  const tf::buffer<Index> &patch_of_facet,
                                  std::size_t n_ids,
                                  tf::buffer<Index> &patch_of_id,
                                  tet_claim_words &claims) -> void {
  if constexpr (tf::core::verification)
    tf::core::verify(patch_of_facet.size() < std::size_t(tet_unowned),
                     "every facet names itself in a claim word");
  claims.allocate_and_initialize(n_ids, tet_unowned);
  tf::parallel_for_each(
      tf::make_sequence_range(patch_of_facet.size()),
      [&ids_of, &claims](std::size_t f) {
        ids_of(f, [&claims, f](Index id) {
          lower_tet_claim_word(claims[std::size_t(id)], tet_claim_word(f));
        });
      },
      tf::checked);
  patch_of_id.allocate(n_ids);
  tf::parallel_for_each(
      tf::make_sequence_range(n_ids),
      [&claims, &patch_of_facet, &patch_of_id](std::size_t id) {
        const tet_claim_word f = claims[id].load(std::memory_order_relaxed);
        patch_of_id[id] =
            f == tet_unowned ? Index(-1) : patch_of_facet[std::size_t(f)];
      },
      tf::checked);
}

} // namespace tf::topology::cdt::dt3
