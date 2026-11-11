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

/// What segment recovery made of one protected segment. A refused segment
/// keeps the subsegments it reached; the complex stays valid.
enum class tet_segment_status : std::uint8_t {
  pending,    ///< Still in recovery; never published.
  recovered,  ///< Every subsegment is an edge of the complex.
  depth,      ///< A subsegment stayed absent or encroached with no exact
              ///< split left.
  degenerate, ///< The segment's two ends are one site.
  wrapped,    ///< Every split it proposed wrapped a protected facet.
  capacity,   ///< The index type cannot name the sites it needs.
  budget,     ///< The sites its splits need would pass the call's site
              ///< budget.
  passes,     ///< A walk of it spent the turns a walk may take
              ///< (@ref tet_execution_tuning::recovery_walk_passes).
  retired     ///< No face holds it any more; its chain is no constraint.
};

} // namespace tf::topology::cdt::dt3
