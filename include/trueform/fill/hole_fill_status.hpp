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
  refused_resource,          ///< The index type cannot name the complex.
  refused_rim_split          ///< The rim fills only by splitting one of its
                             ///< original edges, which the configuration
                             ///< forbids.
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
/// @brief How the chart tier read a group's hole.
///
/// The chart reads the hole as a height field over the plane its rim spans.
/// `admitted` and `rim_plane` hold the patch; every other value is a refusal,
/// which hands the rim on to the table tier and stays stated here, whatever
/// that tier then makes of it. Where a lid was tried, the refusal is the
/// lid's own.
enum class hole_chart_status : std::uint8_t {
  not_attempted, ///< No chart was read: the group was rejected as
                 ///< overlapping another, preflight refused it, or the tier
                 ///< before it held the rim.
  admitted,      ///< The chart holds the group's patch, continuing the host.
  rim_plane,     ///< The chart refused the host, and the rim lies exactly in
                 ///< one plane: the patch is the lid in that plane, planar to
                 ///< the rounding of its placement — a floating lift, then
                 ///< the cast to the mesh's coordinate type.
  no_area,       ///< The rim states no area turned against the chart's normal.
  steep,         ///< A face at the rim leans past seventy degrees from the
                 ///< chart.
  folded,        ///< A face at the rim projects against the patch's own
                 ///< orientation.
  collar_arity,  ///< A face within two rings of the rim is not a triangle.
  uncovered,     ///< The trend fitted to the collar does not cover the rim,
                 ///< its first ring or a sample, or stands steeper than the
                 ///< chart admits there.
  unsampled,     ///< The rim does not triangulate on the chart's lattice.
  unsolved,      ///< The residual solve refused: its operator states a
                 ///< degenerate triangle or is not definite.
  forbidden_chord, ///< The chart's triangulation joins two rim vertices the
                   ///< host already joins.
  resource,        ///< The index type cannot name the chart's samples.
  out_of_lattice   ///< The collar reaches farther from its rim than the
                   ///< mesh's lattice states along the chart's axes.
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
