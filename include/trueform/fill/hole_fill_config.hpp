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

namespace tf {

/// @ingroup fill
/// @brief How one @ref tf::fill_holes run shapes the patches it states.
///
/// The two shaping requests reach every coarse patch alike; the work bounds
/// itself. A chart patch reads neither: it is sampled at the chart's own
/// quality floor and its residual solve is its smoothness.
///
/// `split_rim` states whether the fill may touch the host's original edges
/// at all. Where it may, a tier splits a rim edge where its patch needs one —
/// the table to make the edge part of its complex, the chart where its
/// sampling quality asks — and every split is published as a dyadic record
/// in @ref tf::hole_fill_result's split table, whose carrying faces
/// @ref tf::make_filled_mesh subdivides by the same records, so the stitched
/// mesh is watertight. Where it may not, no original edge is touched: the
/// chart samples against its rim as it stands, and a rim the table could
/// only fill by splitting refuses as `refused_rim_split`.
struct hole_fill_config {
  /// The quality floor a coarse patch is refined toward, in
  /// @ref tf::triangle_quality units. Zero asks for no refinement.
  double min_quality = 0.3;
  /// Whether a refined patch is then faired against its own host
  /// neighbourhood. Only the points a patch minted strictly inside itself
  /// ever move.
  bool fairing = true;
  /// Whether a tier may split the host's original rim edges.
  bool split_rim = true;
};

} // namespace tf
