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
#include "../core/algorithm/generic_generate.hpp"
#include "../core/blocked_buffer.hpp"
#include "./types/simple_intersections.hpp"

namespace tf {

/// @ingroup intersect_types
/// @brief Convert scalar field intersection data to edge connectivity.
///
/// Extracts edge pairs from scalar field intersection data for curve construction.
/// Used internally by @ref tf::make_isocontours.
///
/// @tparam Index The index type.
/// @tparam RealT The coordinate type.
/// @tparam Dims The number of dimensions.
/// @param intersections The @ref tf::scalar_field_intersections data.
/// @return A @ref tf::blocked_buffer of edge pairs.
template <typename Index, typename RealT, std::size_t Dims>
auto make_intersection_edges(
    const tf::intersect::simple_intersections<Index, RealT, Dims>
        &intersections) {
  tf::blocked_buffer<Index, 2> buffer;
  tf::generic_generate(intersections.intersections(), buffer.data_buffer(),
                       [&](const auto &r, auto &buff) {
                         // scalar field intersections either have 1
                         // or 2 elements.
                         if (r.size() != 2)
                           return;
                         buff.push_back(r[0].id);
                         buff.push_back(r[1].id);
                       });
  return buffer;
}
} // namespace tf
