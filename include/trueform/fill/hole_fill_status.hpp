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

/// @ingroup fill
/// @brief What one hole group's fill produced.
///
/// Every refusal publishes an empty patch, no minted points and no splits,
/// and names the rim edge it refused on in @ref tf::hole_fill_result's
/// `offending` lane.
///
/// @see tf::fill_holes
enum class hole_fill_status : std::uint8_t {
  filled,              ///< The group holds a patch.
  refused_open,        ///< A rim of the group does not close.
  refused_invalid_rim, ///< The rim is not a simple cycle of triangle-carried
                       ///< edges, or a protection split pinched it.
  refused_host_face,   ///< A carrying triangle states no usable plane.
  refused_protection_depth, ///< A rim edge stayed absent from the complex
                             ///< at @ref tf::hole_split_depth.
  refused_table,             ///< No admissible triangulation of the rim.
  refused_resource           ///< The index type cannot name the complex.
};

/// @ingroup fill
/// @brief How a group's patch left the refinement that shaped it.
///
/// A tier that does not refine reads `not_attempted`. The three terminal
/// reasons all retain the patch.
enum class hole_refine_status : std::uint8_t {
  not_attempted,   ///< The tier that filled this group does not refine.
  floor_met,       ///< The quality floor holds over the whole patch.
  stalled,         ///< Work remains that no admitted move can resolve.
  budget_exhausted ///< The refinement stopped on its own point budget.
};

/// @ingroup fill
/// @brief How a group's patch left fairing.
enum class hole_fair_status : std::uint8_t {
  not_attempted,  ///< Fairing did not run on this group.
  faired,         ///< The patch carries the faired positions.
  fairing_off,    ///< Fairing was not asked for.
  fairing_refused ///< The stencil or the solve refused the group.
};

} // namespace tf
