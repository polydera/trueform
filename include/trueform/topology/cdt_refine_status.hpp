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

/// @ingroup topology
/// @brief Why a refinement stopped, derived where it returned.
///
/// The target is the one the build clamped and ran against: every triangle
/// of a refined region at that quality floor, and, where encroached
/// constraints may be split, no obligation left that the refinement would
/// still take. Reaching it wins over every other reading, including at the
/// point budget's own boundary. A candidate skipped along the way is not a
/// terminal fact; only the state the refinement ends in is.
///
/// @see tf::cdt_refiner::refine_status()
enum class cdt_refine_status : std::uint8_t {
  floor_met,       ///< The target holds on the triangulation that stands.
  stalled,         ///< Work remains that no admitted split resolves.
  budget_exhausted ///< The refinement stopped on its own point budget.
};

} // namespace tf
