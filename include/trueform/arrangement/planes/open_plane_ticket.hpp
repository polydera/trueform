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

#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/buffer.hpp"

#include <cstddef>

namespace tf::arrangement {

/// A dense `-1`-sentinel table not yet open — a plane's block ticket or a
/// group's route — is opened over `extent` ids, every entry `-1`; an open
/// one stands as it is.
template <typename Index>
auto open_plane_ticket(tf::buffer<Index> &ticket, std::size_t extent) -> void {
  if (ticket.size() != 0)
    return;
  ticket.allocate(extent);
  tf::parallel_fill(ticket, Index(-1));
}

} // namespace tf::arrangement
