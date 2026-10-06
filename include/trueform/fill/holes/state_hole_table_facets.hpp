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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/checked.hpp"
#include "../../core/polygons.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../topology/face_membership_like.hpp"
#include "./hole_chord_is_forbidden.hpp"
#include "./hole_metric.hpp"
#include "./hole_rim.hpp"
#include "./hole_table.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace tf::fill {

/// Read every gathered facet: whether the patch may use it at all, and the
/// plane, area and host angles it carries if it may.
///
/// The reading is the facet's own, so it is taken at the facet's own slot and
/// the retained ones are gathered afterwards. A rim subedge is the intentional
/// exception the guard is never asked about.
template <typename Policy, typename MembershipPolicy, typename Index,
          typename Int, typename RealT>
auto state_hole_table_facets(
    const tf::polygons<Policy> &polygons,
    const tf::face_membership_like<MembershipPolicy> &fm,
    const tf::fill::hole_rim<Index, Int, RealT> &rim, Index n,
    tf::fill::hole_table<Index> &table) -> void {
  const std::size_t found = table.facets.size();
  table.normal.allocate(found);
  table.area.allocate(found);
  table.boundary.allocate(found);
  table.retained.allocate(found);

  const auto is_rim_edge = [n](Index i, Index j) {
    return j == i + 1 || (i == 0 && j == n - 1);
  };
  const auto position = [&rim](Index k) {
    return rim.positions[std::size_t(k)].template as<double>();
  };
  tf::parallel_for_each(
      tf::make_sequence_range(found),
      [&polygons, &fm, &rim, &table, is_rim_edge, position](std::size_t f) {
        table.retained[f] = 0;
        const auto facet = table.facets[f];
        const Index p = facet[0], q = facet[1], r = facet[2];
        if (!is_rim_edge(p, q) &&
            tf::fill::hole_chord_is_forbidden(polygons, fm, rim, p, q))
          return;
        if (!is_rim_edge(q, r) &&
            tf::fill::hole_chord_is_forbidden(polygons, fm, rim, q, r))
          return;
        if (!is_rim_edge(p, r) &&
            tf::fill::hole_chord_is_forbidden(polygons, fm, rim, p, r))
          return;
        const int first = tf::fill::hole_canonical_rotation(
            rim.corners[std::size_t(r)], rim.corners[std::size_t(q)],
            rim.corners[std::size_t(p)]);
        const std::array<Index, 3> wound{r, q, p};
        const auto normal = tf::fill::hole_oriented_normal(
            position(wound[std::size_t(first)]),
            position(wound[std::size_t((first + 1) % 3)]),
            position(wound[std::size_t((first + 2) % 3)]));
        if (!tf::fill::hole_normal_is_usable(normal))
          return;
        const double area = tf::fill::hole_triangle_area(normal);
        double boundary = 0.0;
        if (q == p + 1)
          boundary = std::max(boundary,
                              tf::fill::hole_dihedral_angle(
                                  normal, rim.edges[std::size_t(p)].normal));
        if (r == q + 1)
          boundary = std::max(boundary,
                              tf::fill::hole_dihedral_angle(
                                  normal, rim.edges[std::size_t(q)].normal));
        if (!std::isfinite(area) || !std::isfinite(boundary))
          return;
        table.normal[f] = normal;
        table.area[f] = area;
        table.boundary[f] = boundary;
        table.retained[f] = 1;
      },
      tf::checked);
}

} // namespace tf::fill
