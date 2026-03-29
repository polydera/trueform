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
#include "./construct/make_segment_arrangements.hpp"
#include "./dispatch/segment.hpp"

namespace tf {

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
template <typename Policy>
auto make_segment_arrangements(const tf::segments<Policy> &_segments) {
  return cut::dispatch::segment_dispatch(
      _segments, [](const auto &segments) {
        using Index = std::decay_t<decltype(segments.edges()[0][0])>;
        using RealType = tf::coordinate_type<std::decay_t<decltype(segments)>>;
        constexpr auto Dims =
            tf::coordinate_dims_v<std::decay_t<decltype(segments)>>;

        tf::intersections_within_segments<Index, RealType, Dims> si;
        si.build(segments);

        tf::intersect::segment_intersection_graph<Index, RealType, Dims> sig;
        sig.build(si, segments);

        return tf::cut::make_segment_arrangements(sig, segments,
                                                   si.converter());
      });
}

} // namespace tf
