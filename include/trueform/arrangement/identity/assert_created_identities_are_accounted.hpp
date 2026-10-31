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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/config/verify.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./account_created_identities.hpp"
#include "./identity_substrate.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <cstddef>

namespace tf::arrangement {

/// THE IDENTITY SPACE'S COMPLETENESS, read off a complete plane product:
/// every created identity the substrate holds — the world's, then the plane
/// tier's — is live, retired or vanished (@ref account_created_identities),
/// a death derived there, at both tiers, as what nothing references.
/// The rows are what each plane resolves through, the world's for a plane
/// that reads it and its local block otherwise. A verification check; its
/// caller states that, and that the product holds every carrier.
template <typename Index, typename Int, typename World, typename Arrangement>
auto assert_created_identities_are_accounted(
    const identity_substrate<Index, Int> &identities, const World &world,
    const Arrangement &arrangement) -> void {
  const auto blocks = arrangement.plane_tickets();
  tf::buffer<Index> named;
  tf::generic_generate(
      tf::make_sequence_range(arrangement.n_planes()), named,
      [&](Index plane, tf::buffer<Index> &out) {
        const auto take = [&out](const auto &def) {
          if (def.point_tag_0 < 0)
            out.push_back(def.point_0);
          if (def.point_tag_1 < 0)
            out.push_back(def.point_1);
        };
        const auto block =
            blocks.size() == 0 ? Index(-1) : blocks[std::size_t(plane)];
        if (block != Index(-1)) {
          for (const auto &def : arrangement.current_plane_defs(world, block))
            take(def);
          return;
        }
        tf::core::verify(plane < world.n_planes(),
                         "a plane that reads the world is the world's");
        const auto defs = world.edge_defs();
        for (const auto row : world.plane_edges(plane))
          take(defs[std::size_t(row)]);
      });
  tbb::parallel_sort(named.begin(), named.end());
  named.erase_till_end(std::unique(named.begin(), named.end()));

  const auto accounts = account_created_identities(
      identities, arrangement.triangles(), named, arrangement.n_flat_points(),
      Index(arrangement.n_base_created() + identities.n_plane_created()));
  tf::parallel_for_each(
      accounts,
      [](created_identity_account account) {
        tf::core::verify(account != created_identity_account::emitted_dead,
                         "a triangle names a retired identity");
        tf::core::verify(account != created_identity_account::unaccounted,
                         "a created identity is neither live, retired nor "
                         "vanished");
      },
      tf::checked);
}

} // namespace tf::arrangement
