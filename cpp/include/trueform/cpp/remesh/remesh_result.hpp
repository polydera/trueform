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

#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <cstdint>

namespace tf::cpp {

/// @brief What a remesh hands back: the new mesh and the region each face
/// came from.
template <typename Index, typename Real> struct remesh_result {
  static_assert(is_supported_index_v<Index>,
                "remesh Index must be an unqualified supported index type");

  tf::polygons_buffer<Index, Real, 3, 3> mesh;
  nd_array<std::int32_t> regions;
};

} // namespace tf::cpp
