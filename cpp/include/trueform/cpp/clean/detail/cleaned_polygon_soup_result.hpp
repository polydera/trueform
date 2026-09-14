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

#include "trueform/cpp/clean/supported.hpp"
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/core/segments_buffer.hpp"

#include <cstddef>
#include <type_traits>

namespace tf::cpp::detail {

/// The carrier a soup of a given arity cleans into: two-vertex soups are edge
/// meshes and three-vertex soups are triangle meshes. An unsupported spelling
/// names no type, so it disappears under detection rather than hard-erroring.
template <typename Index, typename Real, std::size_t Dims, std::size_t Vertices,
          typename = void>
struct basic_cleaned_polygon_soup_result {};

template <typename Index, typename Real, std::size_t Dims>
struct basic_cleaned_polygon_soup_result<
    Index, Real, Dims, 2,
    std::enable_if_t<is_supported_cleaned_polygon_soup_v<Index, Real, Dims, 2>>> {
  using type = tf::segments_buffer<Index, Real, Dims>;
};

template <typename Index, typename Real, std::size_t Dims>
struct basic_cleaned_polygon_soup_result<
    Index, Real, Dims, 3,
    std::enable_if_t<is_supported_cleaned_polygon_soup_v<Index, Real, Dims, 3>>> {
  using type = tf::polygons_buffer<Index, Real, Dims, 3>;
};

} // namespace tf::cpp::detail
