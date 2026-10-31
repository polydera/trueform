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
#include "./hole_chord_is_forbidden.hpp"
#include "./hole_metric.hpp"
#include "tbb/parallel_sort.h"
#include <array>
#include <cstddef>
#include <limits>

namespace tf::fill {

/// State a chart patch in flat identities.
///
/// Rim position `k` is `flat[k]` — the rim's own vertex, or a split the rim
/// was extended by — and sample `j` is the group's mint
/// `n_points + split_points.size() + j`, so the patch stitches to the host
/// by identity, the splits are minted first and the samples after them, all
/// verbatim. A chord between two rim positions faces the same guard every
/// other tier's chords do, each end named with the carrier it inherits; the
/// rim's own edges are the stitch and are never asked about. The triangles
/// leave in the canonical order — each rotated onto its smallest corner with
/// its winding kept, the block then sorted.
///
/// `triangles` and `minted` are appended to only when the patch holds.
template <typename Policy, typename MembershipPolicy, typename Index,
          typename RealT>
auto emit_hole_chart_patch(const tf::polygons<Policy> &polygons,
                           const tf::face_membership_like<MembershipPolicy> &fm,
                           const tf::buffer<Index> &flat,
                           const tf::buffer<Index> &carrier, Index n_points,
                           const tf::buffer<tf::point<RealT, 3>> &split_points,
                           const tf::buffer<tf::point<RealT, 3>> &points,
                           const tf::buffer<std::array<Index, 3>> &positioned,
                           tf::buffer<std::array<Index, 3>> &triangles,
                           tf::buffer<tf::point<RealT, 3>> &minted)
    -> tf::hole_chart_status {
  const std::size_t n = flat.size();
  if (split_points.size() + points.size() >
      std::size_t(std::numeric_limits<Index>::max() - n_points))
    return tf::hole_chart_status::resource;
  const Index first_sample = Index(n_points + Index(split_points.size()));

  const auto adjacent = [n](std::size_t a, std::size_t b) {
    return b == (a + 1 == n ? 0 : a + 1) || a == (b + 1 == n ? 0 : b + 1);
  };
  const auto faces = polygons.faces();
  for (const auto &triangle : positioned)
    for (std::size_t side = 0; side < 3; ++side) {
      const std::size_t a = std::size_t(triangle[side]);
      const std::size_t b = std::size_t(triangle[(side + 1) % 3]);
      if (a < n && b < n && !adjacent(a, b) &&
          tf::fill::hole_chord_is_forbidden(faces, fm, n_points, flat[a],
                                            carrier[a], flat[b], carrier[b]))
        return tf::hole_chart_status::forbidden_chord;
    }

  const auto named = [&flat, n, first_sample](Index position) {
    return std::size_t(position) < n
               ? flat[std::size_t(position)]
               : Index(first_sample + Index(std::size_t(position) - n));
  };
  const std::size_t first = triangles.size();
  for (const auto &triangle : positioned) {
    const std::array<Index, 3> corners{named(triangle[0]), named(triangle[1]),
                                       named(triangle[2])};
    const int root =
        tf::fill::hole_canonical_rotation(corners[0], corners[1], corners[2]);
    triangles.push_back({corners[std::size_t(root)],
                         corners[std::size_t((root + 1) % 3)],
                         corners[std::size_t((root + 2) % 3)]});
  }
  tbb::parallel_sort(triangles.begin() + first, triangles.end());
  for (const auto &point : split_points)
    minted.push_back(point);
  for (const auto &point : points)
    minted.push_back(point);
  return tf::hole_chart_status::admitted;
}

} // namespace tf::fill
