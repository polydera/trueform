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
#include "./records/for_each_cut_chord.hpp"
#include "./records/simple_intersections.hpp"

namespace tf {

/// @ingroup intersect_curves
/// @brief Convert scalar field intersection data to edge connectivity.
///
/// Extracts edge pairs from scalar field intersection data for curve
/// construction. Used by @ref tf::make_isocontours and the return_curves
/// overloads of the isoband functions.
///
/// @param intersections The scalar-field crossings of one build.
/// @param faces The faces those crossings were stated on.
/// @return A @ref tf::blocked_buffer of index pairs, one per chord.
template <typename Index, typename RealT, std::size_t Dims, typename Faces>
auto make_intersection_edges(
    const tf::intersect::simple_intersections<Index, RealT, Dims>
        &intersections,
    const Faces &faces) {
  auto ipts = intersections.intersection_points();
  tf::blocked_buffer<Index, 2> buffer;
  tf::generic_generate(
      intersections.intersections(), buffer.data_buffer(),
      [&](const auto &r, auto &buff) {
        tf::intersect::for_each_cut_group(
            r, static_cast<Index>(faces[r[0].object].size()), ipts,
            [&](Index id0, Index id1, bool /*on_boundary*/) {
              buff.push_back(id0);
              buff.push_back(id1);
            });
      });
  return buffer;
}
} // namespace tf
