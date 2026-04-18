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
#include "./delaunay_flipper.hpp"
#include "./ear_cutter.hpp"

namespace tf {

/// @ingroup topology
/// @brief Constrained Delaunay triangulation of 2D polygons.
///
/// Combines ear clipping with Lawson edge flips to produce a
/// constrained Delaunay triangulation. Same API as `tf::ear_cutter`.
/// Polygon boundary edges (consecutive index pairs) are constrained.
///
/// @tparam Index The index type.
/// @tparam Int The integer type for exact predicates (default: int32).
template <typename Index, typename Int = tf::exact::int32>
class delaunay_triangulator {
public:
  auto faces() const { return _earcut.faces(); }
  auto indices_buffer() const -> const tf::buffer<Index> & {
    return _earcut.indices_buffer();
  }
  auto indices_buffer() -> tf::buffer<Index> & {
    return _earcut.indices_buffer();
  }

  auto clear() -> void {
    _earcut.clear();
    _flipper.clear();
  }

  template <typename Policy>
  auto build(const tf::points<Policy> &points) -> bool {
    bool ok = _earcut.build(points);
    auto n = static_cast<Index>(points.size());
    _flipper.flip(
        tf::make_faces<3>(_earcut.indices_buffer()), points,
        [n](Index a, Index b) -> bool {
          auto d = a > b ? a - b : b - a;
          return d == 1 || d == n - 1;
        });
    return ok;
  }

private:
  tf::ear_cutter<Index, Int> _earcut;
  tf::delaunay_flipper<Index, Int> _flipper;
};

} // namespace tf
