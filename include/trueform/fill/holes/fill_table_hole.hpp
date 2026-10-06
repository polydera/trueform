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
#include "./hole_rim.hpp"
#include "./hole_table.hpp"
#include "./protect_hole_rim.hpp"
#include "./solve_hole_table.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::fill {

/// Fill a rim through the tetrahedralization of its own sites.
///
/// The rim is protected first — every rim edge becomes an edge of the
/// complex, at the cost of dyadic splits the whole pipeline shares — and
/// only then are its split positions NAMED, in the canonical order of the
/// split records themselves, so a mint's identity is fixed before any
/// numerical reading of the patch uses it. The complex's facets are the
/// candidates the two-stage table optimizes over.
template <typename Policy, typename MembershipPolicy, typename Index,
          typename Int, typename RealT>
auto fill_table_hole(const tf::polygons<Policy> &polygons,
                     const tf::face_membership_like<MembershipPolicy> &fm,
                     Index n_points,
                     tf::fill::hole_table_scratch<Index, Int, RealT> &complex,
                     tf::fill::hole_table<Index> &table,
                     tf::fill::hole_rim<Index, Int, RealT> &rim,
                     tf::buffer<std::array<Index, 3>> &triangles,
                     tf::buffer<tf::point<RealT, 3>> &minted,
                     tf::buffer<tf::hole_split<Index>> &splits,
                     Index &offending) -> tf::hole_fill_status {
  const auto protection =
      tf::fill::protect_hole_rim(polygons.points(), complex, rim, offending);
  if (protection != tf::hole_fill_status::filled)
    return protection;

  complex.split_positions.clear();
  for (Index k = 0; k < Index(rim.size()); ++k)
    if (tf::fill::hole_position_is_split(rim, k))
      complex.split_positions.push_back(k);
  std::sort(complex.split_positions.begin(), complex.split_positions.end(),
            [&rim](Index x, Index y) {
              const auto &a = rim.edges[std::size_t(x)];
              const auto &b = rim.edges[std::size_t(y)];
              if (a.v0 != b.v0)
                return a.v0 < b.v0;
              if (a.v1 != b.v1)
                return a.v1 < b.v1;
              return a.t0 < b.t0;
            });
  for (std::size_t mint = 0; mint < complex.split_positions.size(); ++mint) {
    const Index k = complex.split_positions[mint];
    const auto &edge = rim.edges[std::size_t(k)];
    rim.corners[std::size_t(k)] = Index(n_points + Index(mint));
    minted.push_back(rim.positions[std::size_t(k)]);
    splits.push_back({edge.face, edge.v0, edge.v1,
                      rim.corners[std::size_t(k)], edge.t0});
  }

  const auto solved =
      tf::fill::solve_hole_table(polygons, fm, rim, complex, table, triangles);
  if (solved != tf::hole_fill_status::filled) {
    triangles.clear();
    minted.clear();
    splits.clear();
  }
  return solved;
}

} // namespace tf::fill
