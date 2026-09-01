/*
 * Copyright (c) 2025 XLAB
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

/// @ingroup topology
/// @brief What a constrained Delaunay build's region labels are.
///
/// The walls are the boundary constraints; a plain constraint is preserved
/// but re-entered, so a slit never opens a region.
///
/// @see tf::constrained_delaunay_triangulator::region_labels()
enum class cdt_region_mode {
  nesting,   ///< Parity of the walls one path from the outside crosses.
  components ///< Id of the component the walls cut, 0 the hull exterior.
};

} // namespace tf
