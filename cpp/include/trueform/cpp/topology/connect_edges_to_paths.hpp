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

#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"

#include <cstdint>

namespace tf::cpp {
/// @brief Assemble edges into continuous paths.
///
/// The carriers name no index of their own, so the ENTRY is what states which
/// widths this build answers for: an index the matrix left out has no overload,
/// and naming it fails where it is called.
#define TF_CPP_DECLARE_CONNECT_EDGES_TO_PATHS(Index)                           \
  auto connect_edges_to_paths(const nd_array<Index> &edges)                    \
      ->offset_blocked_buffer<Index, Index>

TF_CPP_MATRIX_FOR_EACH_INDEX(TF_CPP_DECLARE_CONNECT_EDGES_TO_PATHS)

#undef TF_CPP_DECLARE_CONNECT_EDGES_TO_PATHS

} // namespace tf::cpp
