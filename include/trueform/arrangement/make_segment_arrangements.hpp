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
#include "../core/resolved_output_real.hpp"
#include "../exact/resolve_int_type.hpp"
#include "./construct/make_segment_arrangements.hpp"
#include "./operands/segment.hpp"

namespace tf {

namespace arrangement {

template <typename Int, typename OutputCoordinateType, typename Segments>
auto segment_arrangements_impl(const Segments &segments) {
  using Index = std::decay_t<decltype(segments.edges()[0][0])>;
  using InputReal = tf::coordinate_type<Segments>;
  using ResolvedInt = tf::exact::resolve_int_type<Int, InputReal>;
  using PipelineReal =
      std::conditional_t<std::is_integral_v<InputReal>, InputReal, double>;
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType, InputReal>;
  constexpr auto Dims = tf::coordinate_dims_v<Segments>;

  tf::intersections_within_segments<Index, PipelineReal, Dims, ResolvedInt> si;
  si.build(segments);

  tf::intersect::segment_intersection_graph<Index, Dims, ResolvedInt> sig;
  sig.build(si, segments);

  auto result =
      tf::arrangement::make_segment_arrangements<OutputCoordinateType>(
          sig, segments, si.converter());

  if constexpr (!std::is_integral_v<InputReal> &&
                std::is_integral_v<RealOut>) {
    auto conv = si.converter();
    return std::tuple_cat(std::make_tuple(std::move(result.first),
                                          std::move(result.second)),
                          std::make_tuple(std::move(conv)));
  } else {
    return result;
  }
}

} // namespace arrangement

/// @ingroup arrangement_planar
/// @brief Split segments at all intersection points.
///
/// Computes all pairwise segment intersections, merges coincident
/// points, and returns the subdivided segments with edge labels
/// mapping each sub-edge to its original edge.
///
/// @tparam Policy The policy type for the segments.
/// @param _segments The input segments (plain or with tree/edge_membership).
/// @return A pair of (segments_buffer, edge_labels).
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy>
auto make_segment_arrangements(const tf::segments<Policy> &_segments) {
  return arrangement::dispatch::segment_dispatch(
      _segments, [](const auto &segments) {
        return arrangement::segment_arrangements_impl<Int,
                                                      OutputCoordinateType>(
            segments);
      });
}

} // namespace tf
