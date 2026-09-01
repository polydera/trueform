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

#include "./plane_recovery_name.hpp"

namespace tf::arrangement {

/// `tier` says which table indexes `root`: `0` the original table, `1` the
/// local one. An edge is an edge — the pair (tier, id) is its whole name, and
/// every consumer resolves it through the one lookup.
template <typename Index, typename Param> struct plane_recovery_statement {
  plane_recovery_name<Index, Param> name;
  Index root, plane, tier;
};

} // namespace tf::arrangement
