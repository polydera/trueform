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

/// Serial liveness and hull classification; concurrent insertion publishes this
/// lane after joining, from temporary ownership and the infinity corner.
enum class tet_state : std::uint8_t { live, infinite, dead };

} // namespace tf::topology::cdt::dt3
