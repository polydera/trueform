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

#include "trueform/cpp/clean/detail/cleaned_polygon_soup_result.hpp"
#include "trueform/cpp/clean/supported.hpp"
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <cstddef>
#include <type_traits>

namespace tf::cpp {

/// @brief The carrier a fixed-arity soup cleans into.
///
/// Two-vertex soups become edge meshes and three-vertex soups become triangle
/// meshes. The result owns detached connectivity and point storage.
template <typename Index, typename Real, std::size_t Dims, std::size_t Vertices>
using basic_cleaned_polygon_soup_result =
    typename detail::basic_cleaned_polygon_soup_result<Index, Real, Dims,
                                                       Vertices>::type;

/// @brief Clean a typed 2D or 3D segment/triangle soup into indexed geometry.
///
/// The input must have shape [Vertices, Dims] or [N, Vertices, Dims], where
/// Vertices is 2 or 3. Index controls the returned connectivity width. A soup
/// names its vertices once each and references all of them, so both
/// configuration booleans are accepted and ignored. Zero selects exact
/// cleaning; negative and nonfinite tolerances are rejected.
template <typename Index = default_index_t, typename Real, std::size_t Dims = 3,
          std::size_t Vertices = 3>
auto cleaned_polygon_soup(const nd_array<Real> &polygons,
                          Real tolerance = Real{},
                          bool remove_duplicate_primitives = true,
                          bool remove_unreferenced_points = true)
    -> basic_cleaned_polygon_soup_result<Index, Real, Dims, Vertices>;

// Preserve the compiled template signature while making unsupported explicit
// spellings disappear under C++17 detection.
template <typename Index, typename Real, std::size_t Dims, std::size_t Vertices,
          std::enable_if_t<
              !is_supported_cleaned_polygon_soup_v<Index, Real, Dims, Vertices>,
              int> = 0>
auto cleaned_polygon_soup(const nd_array<Real> &, Real = Real{}, bool = true,
                          bool = true) -> void = delete;

#define TF_CPP_EXTERN_CLEAN_SOUP(Index, Real, Dims)                            \
  extern template auto cleaned_polygon_soup<Index, Real, Dims, 2>(             \
      const nd_array<Real> &, Real, bool, bool)                                \
      -> basic_cleaned_polygon_soup_result<Index, Real, Dims, 2>;              \
  extern template auto cleaned_polygon_soup<Index, Real, Dims, 3>(             \
      const nd_array<Real> &, Real, bool, bool)                                \
      -> basic_cleaned_polygon_soup_result<Index, Real, Dims, 3>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS(TF_CPP_EXTERN_CLEAN_SOUP)

#undef TF_CPP_EXTERN_CLEAN_SOUP

} // namespace tf::cpp
