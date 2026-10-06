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

/// @ingroup topology
/// @brief Why a Delaunay tetrahedralization holds no cells.
///
/// @see tf::delaunay_tetrahedralizer::refusal()
enum class tetrahedralization_refusal {
  none,           ///< The build completed.
  rank_deficient, ///< The welded sites span a plane, a line or a point.
  index_capacity  ///< The cells outgrew the index type that names them.
};

} // namespace tf
