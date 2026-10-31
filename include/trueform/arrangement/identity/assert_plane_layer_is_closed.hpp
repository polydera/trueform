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

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/checked.hpp"
#include "../../core/config/verify.hpp"
#include "./identity_substrate.hpp"

#include <array>

namespace tf::arrangement {

/// Invariant I read off a published plane product (@ref identity_substrate):
/// every corner of `triangles` below the plane tier's own created extent —
/// `n_flat` originals, then `base_created` local identities — is a key `L`
/// does not retire, so the plane layer alone answered it `P∘L`. A
/// verification check; its caller states that.
template <typename Index, typename Int, typename Triangles>
auto assert_plane_layer_is_closed(
    const identity_substrate<Index, Int> &identities,
    const Triangles &triangles, Index n_flat, Index base_created) -> void {
  tf::parallel_for_each(
      triangles,
      [&](const std::array<Index, 3> &triangle) {
        for (const auto corner : triangle) {
          if (corner >= n_flat + base_created)
            continue;
          const auto target =
              corner < n_flat
                  ? identities.local_target(Index(0), corner)
                  : identities.local_target(Index(1), corner - n_flat);
          tf::core::verify(target == Index(-1),
                           "a plane product holds a key the local layer "
                           "retired");
        }
      },
      tf::checked);
}

} // namespace tf::arrangement
