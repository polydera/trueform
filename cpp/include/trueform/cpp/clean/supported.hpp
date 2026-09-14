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

#include "trueform/cpp/core/index_type.hpp"

#include <cstddef>
#include <type_traits>

namespace tf::cpp {

/// @brief Whether raw point-array cleaning has a compiled specialization.
template <typename Real, std::size_t Dims>
inline constexpr bool is_supported_cleaned_points_v =
    (std::is_same_v<Real, float> || std::is_same_v<Real, double>) &&
    (Dims == 2 || Dims == 3);

/// @brief Whether typed polygon-soup cleaning has a compiled specialization.
template <typename Index, typename Real, std::size_t Dims, std::size_t Vertices>
inline constexpr bool is_supported_cleaned_polygon_soup_v =
    is_supported_cleaned_points_v<Real, Dims> && is_supported_index_v<Index> &&
    std::is_same_v<Index, std::remove_cv_t<Index>> &&
    (Vertices == 2 || Vertices == 3);

} // namespace tf::cpp
