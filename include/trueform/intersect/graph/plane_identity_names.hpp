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

#include "../../core/algorithm/compute_offsets.hpp"
#include "../../core/buffer.hpp"
#include "../../core/range.hpp"
#include "tbb/parallel_sort.h"
#include <array>
#include <iterator>
#include <tuple>

namespace tf::intersect::graph {

/// A junction's NAME: the original features that define it, never a
/// number. `{0, ibp id}` an intersection point a landing names, `{1,
/// flat vertex}` an original vertex, `{2, P, g, h}` the full plane
/// triple of a seam junction, `{3, u, v, plane}` an original edge cut
/// by a plane, `{4, u0, v0, u1, v1}` two original edges, and `{5, a0,
/// a1, b0, b1}` the two exact generation-input pieces a PA recovery
/// crossing joins, each stated by its ordered flat endpoint pair. Equal
/// names are one point BY CONSTRUCTION.
template <typename Index> using plane_name = std::array<Index, 5>;

template <typename Index> auto plane_point_name(Index id) -> plane_name<Index> {
  return {Index(0), id, Index(-1), Index(-1), Index(-1)};
}

template <typename Index>
auto plane_vertex_name(Index flat) -> plane_name<Index> {
  return {Index(1), flat, Index(-1), Index(-1), Index(-1)};
}

/// One statement of one junction: the name, the canonical edge it cuts,
/// the edge it crosses there, and the plane it was seen in.
template <typename Index> struct plane_name_statement {
  plane_name<Index> name;
  Index root, partner, plane;
};

/// THE CLOSURE — and it is a sort. Equal names are one identity, so the
/// runs of the sorted table ARE the classes and no numeric comparison
/// decides sameness. The order inside a run puts the name's canonical
/// smallest root first, which is where it will be born.
template <typename Index>
auto close_plane_names(tf::buffer<plane_name_statement<Index>> &statements,
                       tf::buffer<Index> &class_offsets) -> Index {
  using statement_t = plane_name_statement<Index>;
  class_offsets.clear();
  if (statements.size() == 0)
    return 0;
  tbb::parallel_sort(statements.begin(), statements.end(),
                     [](const statement_t &x, const statement_t &y) {
                       return std::tie(x.name, x.root, x.partner, x.plane) <
                              std::tie(y.name, y.root, y.partner, y.plane);
                     });
  class_offsets.reserve(statements.size() + 1);
  tf::compute_offsets(statements, std::back_inserter(class_offsets), Index(0),
                      [](const statement_t &x, const statement_t &y) {
                        return x.name == y.name;
                      });
  return Index(class_offsets.size()) - 1;
}

} // namespace tf::intersect::graph
