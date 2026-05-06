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
#include "../exact/resolve_int_type.hpp"
#include "./construct/make_segment_arrangements.hpp"
#include "./dispatch/segment.hpp"

namespace tf {

namespace cut::impl {

template <typename Int, typename OutputCoordinateType, typename Segments>
auto segment_arrangements_impl(const Segments &segments) {
  using Index = std::decay_t<decltype(segments.edges()[0][0])>;
  using InputReal = tf::coordinate_type<Segments>;
  using ResolvedInt = tf::exact::resolve_int_type<Int, InputReal>;
  constexpr auto Dims = tf::coordinate_dims_v<Segments>;

  tf::intersections_within_segments<Index, double, Dims, ResolvedInt> si;
  si.build(segments);

  tf::intersect::segment_intersection_graph<Index, Dims, ResolvedInt> sig;
  sig.build(si, segments);

  return tf::cut::make_segment_arrangements<OutputCoordinateType>(
      sig, segments, si.converter());
}

} // namespace cut::impl

/// @ingroup cut_planar
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
  return cut::dispatch::segment_dispatch(_segments, [](const auto &segments) {
    return cut::impl::segment_arrangements_impl<Int, OutputCoordinateType>(
        segments);
  });
}

} // namespace tf
