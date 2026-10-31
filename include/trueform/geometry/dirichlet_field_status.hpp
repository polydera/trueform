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
/// @brief What a @ref tf::make_dirichlet_field solve answered.
///
/// - `refused_operator`: a corner weight that is not finite, or a point whose
///   lumped mass is not positive — a point no triangle has as a corner, or
///   one whose every triangle is degenerate;
/// - `refused_unanchored`: a connected run of free points that reaches no
///   fixed one;
/// - `refused_index`: an index type too narrow to name the factor;
/// - `refused_indefinite`: a pivot that is not positive or not finite;
/// - `refused_residual`: an answer whose true residual misses the
///   componentwise backward error it is accepted at.
enum class dirichlet_field_status : std::uint8_t {
  solved,
  refused_operator,
  refused_unanchored,
  refused_index,
  refused_indefinite,
  refused_residual
};

} // namespace tf
