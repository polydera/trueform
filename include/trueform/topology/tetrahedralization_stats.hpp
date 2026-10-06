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
#include <cstddef>

namespace tf {

/// @ingroup topology
/// @brief What one Delaunay tetrahedralization cost to build.
///
/// @see tf::delaunay_tetrahedralizer::stats()
struct tetrahedralization_stats {
  /// Sites inserted past the seed's own four.
  std::size_t insertions = 0;
  /// How often the dead slots were swept away.
  std::size_t compactions = 0;
  /// Spatial insertion bands processed concurrently.
  std::size_t parallel_bands = 0;
  /// Sites committed by concurrent streams.
  std::size_t parallel_insertions = 0;
  /// Barriers that transferred free rows before retrying blocked streams.
  std::size_t pool_restarts = 0;
};

} // namespace tf
