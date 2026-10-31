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

namespace tf {

/// @ingroup geometry
/// @brief The operator a @ref tf::make_dirichlet_field vanishes under at
/// every free point: `harmonic` is `L x = 0`, `biharmonic` is
/// `L M^-1 L x = 0`.
enum class dirichlet_order : std::uint8_t { harmonic, biharmonic };

} // namespace tf
