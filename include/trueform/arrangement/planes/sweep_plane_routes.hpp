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
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"

#include <cstddef>

namespace tf::arrangement {

/// THE ROUTER SWEEP: every rewrite of the group router is this one sweep.
/// Each live route takes the caller's verdict once; a parked route (< 0) is
/// never consulted.
template <typename Index, typename Verdict>
auto sweep_plane_routes(tf::buffer<Index> &group_router, Verdict &&verdict)
    -> void {
  tf::parallel_for_each(
      tf::make_sequence_range(group_router.size()),
      [&](std::size_t at) {
        const auto route = group_router[at];
        if (route >= Index(0))
          group_router[at] = verdict(route);
      },
      tf::checked);
}

} // namespace tf::arrangement
