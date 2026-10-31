/*
 * Copyright (c) 2026 XLAB
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
#include "../../core/buffer.hpp"
#include "../../core/point.hpp"
#include "../../core/polygons.hpp"
#include "../../topology/face_membership_like.hpp"
#include "../hole_fill_status.hpp"
#include "../hole_split.hpp"
#include "./emit_hole_chart_patch.hpp"
#include "./hole_rim.hpp"
#include "./hole_rim_is_planar.hpp"
#include "./make_hole_chart_patch.hpp"
#include "./make_hole_lid_patch.hpp"
#include "./name_hole_chart_rim.hpp"
#include <array>

namespace tf::fill {

/// Fill a rim as a height field over the plane it spans.
///
/// The chart continues the host: a plane is the trend whose curvature
/// vanishes, so an exactly planar host fills in its own plane through the
/// same mechanism as a curved one. Where the chart refuses for any reason
/// and the rim lies exactly in one plane, the lid in that plane is the floor
/// under the whole tier. Where `split_rim` allows it a continued patch may
/// split rim edges its sampling asks to; each split is named before any
/// sample (@ref tf::fill::name_hole_chart_rim), minted where its record says
/// and published in `splits`, which the stitched mesh replays into the
/// carrying faces. A lid splits nothing. The residual solve is the patch's
/// smoothness, which is why a chart patch is not faired again.
///
/// Returns `admitted` or `rim_plane` where the chart holds the patch; any
/// other status — the lid's own refusal where a lid was tried — leaves
/// nothing behind and hands the rim on to the table tier.
template <typename Policy, typename MembershipPolicy, typename Index,
          typename Int, typename RealT>
auto fill_chart_hole(const tf::polygons<Policy> &polygons,
                     const tf::face_membership_like<MembershipPolicy> &fm,
                     const tf::fill::hole_rim<Index, Int, RealT> &rim,
                     Index n_points, double unit, bool split_rim,
                     tf::fill::hole_chart_scratch<Index, Int, RealT> &scratch,
                     tf::fill::hole_chart_rim_names<Index, RealT> &names,
                     tf::buffer<std::array<Index, 3>> &triangles,
                     tf::buffer<tf::point<RealT, 3>> &minted,
                     tf::buffer<tf::hole_split<Index>> &splits,
                     tf::hole_refine_status &refined) -> tf::hole_chart_status {
  const auto emit = [&](tf::hole_chart_status made) {
    if (made != tf::hole_chart_status::admitted &&
        made != tf::hole_chart_status::rim_plane)
      return made;
    tf::fill::name_hole_chart_rim(polygons.points(), rim, n_points,
                                  scratch.rim_splits, names);
    const auto emitted = tf::fill::emit_hole_chart_patch(
        polygons, fm, names.flat, names.carrier, n_points, names.points,
        scratch.points, scratch.triangles, triangles, minted);
    if (emitted != tf::hole_chart_status::admitted)
      return emitted;
    for (const auto &record : names.records)
      splits.push_back(record);
    return made;
  };
  const auto continued = emit(tf::fill::make_hole_chart_patch(
      polygons, fm, rim.corners, unit, split_rim, scratch, refined));
  if (continued == tf::hole_chart_status::admitted ||
      !tf::fill::hole_rim_is_planar(rim))
    return continued;
  return emit(tf::fill::make_hole_lid_patch(polygons, rim.corners, unit,
                                            scratch, refined));
}

} // namespace tf::fill
