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
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// What one site's insertion did: its star replaced its cavity, the cell
/// lanes could not take the star, or the cavity held both cells of a
/// protected facet and was given back untouched.
enum class tet_insertion : std::uint8_t { placed, capacity, wrapped };

} // namespace tf::topology::cdt::dt3
