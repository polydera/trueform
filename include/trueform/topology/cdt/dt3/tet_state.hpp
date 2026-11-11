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

/// Liveness and hull classification, the one statement of a dead row: written
/// by whatever writes or kills the row, under concurrent insertion the stream
/// holding it.
enum class tet_state : std::uint8_t { live, infinite, dead };

} // namespace tf::topology::cdt::dt3
