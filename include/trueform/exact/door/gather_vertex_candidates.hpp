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

#include "../../core/vector.hpp"
#include "../meta.hpp"
#include "./placement_tables.hpp"
#include "./quantized_plane.hpp"
#include "./wide_dot.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace tf::exact::door {

/// The most names one vertex may stand on. The cut is made after the
/// sort, so the list is the first this many in name order and two
/// vertices carrying the same names still see the same one. It is what
/// bounds the widest-triple search, which is cubic in the list.
inline constexpr std::size_t candidate_limit = 24;

/// The planes a vertex stands on, deduplicated and in name order, and
/// the mean direction of every face incident to it.
///
/// Name order is what makes a placement a pure function of the names:
/// two vertices of two forms carrying the same faces' names see the
/// same list in the same order, so they are placed on the same integer.
///
/// A name further than the band from the vertex is not one it stands
/// on: ranks 3 and 2 land on their planes and the certificate admits
/// nothing further than the band away, so such a name cannot appear in
/// the answer and only costs the triple search its cube. A face's name
/// is offset at the face's centroid, so on a mesh whose faces are
/// larger than the band this is the whole list, and the smooth vertex
/// reaches its own rank without enumerating anything.
///
/// The direction is the mean over all of them: it is the vertex's own
/// normal, which rank 1 quantizes at the vertex and not at a centroid,
/// so a name too far to stand on still states which way the surface
/// faces.
template <typename Index, typename Int, typename RealType, typename Candidates>
auto gather_vertex_candidates(
    const placement_tables<Index, Int, RealType> &tables, Index flat,
    Int tolerance, Candidates &candidates) -> tf::vector<double, 3> {
  using T2 = typename tf::exact::meta<Int>::T2;
  candidates.clear();
  tf::vector<double, 3> direction{0.0, 0.0, 0.0};
  const auto &point = tables.points[std::size_t(flat)];
  const auto from = std::size_t(tables.incidence_offsets[std::size_t(flat)]);
  const auto to = std::size_t(tables.incidence_offsets[std::size_t(flat) + 1]);
  const auto zero = quantized_plane<Int>{};
  for (auto at = from; at < to; ++at) {
    const auto &plane = tables.planes[std::size_t(tables.incidence[at])];
    if (plane.normal == zero.normal)
      continue;
    const double x = static_cast<double>(plane.normal[0]);
    const double y = static_cast<double>(plane.normal[1]);
    const double z = static_cast<double>(plane.normal[2]);
    const double length = std::sqrt(x * x + y * y + z * z);
    if (!(length > 0.0))
      continue;
    direction[0] += x / length;
    direction[1] += y / length;
    direction[2] += z / length;
    // The band widened by the lattice unit a near-line answer may round
    // by, and by the double evaluation's own margin.
    const double reach =
        (static_cast<double>(tolerance) + 1.0) * length * (1.0 + 1e-12);
    const T2 gap = T2(plane.offset) - wide_dot<Int>(plane.normal, point);
    if (std::abs(static_cast<double>(gap)) > reach)
      continue;
    candidates.push_back(plane);
  }
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  if (candidates.size() > candidate_limit)
    candidates.resize(candidate_limit);
  return direction;
}

} // namespace tf::exact::door
