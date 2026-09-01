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
#include "./unconstrained_delaunay_face_next.hpp"
#include <array>
#include <cstddef>

namespace tf::topology::cdt {

template <typename Index> struct unconstrained_delaunay_triangle_orbit {
  std::array<Index, 3> darts;
  bool closed;
};

/// Trace the first three darts of a left-face orbit and report whether they
/// close. Interior Delaunay faces close; the exterior orbit generally does not.
template <typename Owner>
auto trace_unconstrained_delaunay_triangle(const Owner &owner,
                                           typename Owner::index_type first)
    -> unconstrained_delaunay_triangle_orbit<typename Owner::index_type> {
  unconstrained_delaunay_triangle_orbit<typename Owner::index_type> result{
      {first, first, first}, false};
  for (std::size_t corner = 1; corner < result.darts.size(); ++corner)
    result.darts[corner] =
        unconstrained_delaunay_face_next(owner, result.darts[corner - 1]);
  result.closed = unconstrained_delaunay_face_next(
                      owner, result.darts.back()) == result.darts.front();
  return result;
}

} // namespace tf::topology::cdt
