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

#include "trueform/core/edges.hpp"
#include "trueform/core/faces.hpp"
#include "trueform/core/points.hpp"
#include "trueform/core/static_size.hpp"
#include "trueform/core/views/blocked_range.hpp"
#include "trueform/cpp/core/cache.hpp"
#include "trueform/cpp/core/detail/face_blocks_array.hpp"
#include "trueform/cpp/core/edge_mesh.hpp"
#include "trueform/cpp/core/edge_mesh_cache.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"

#include <cstddef>
#include <utility>

namespace tf::cpp::detail {

/// @brief The mesh a tuple-equivalent entry reads its arrays as.
///
/// A caller that hands arrays rather than an assembly holds no cache, so the
/// entry holds one for the length of the call. The arrays themselves are
/// BORROWED, exactly as any other reading borrows them — the caller's storage
/// outlives the call it made.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon = 3>
class array_mesh {
public:
  using mesh_type = cpp::mesh<Index, Real, Dims, Ngon>;

  array_mesh(const nd_array<Index> &faces, const nd_array<Real> &points)
      : _indices(faces), _points(points) {}
  array_mesh(const offset_blocked_buffer<Index, Index> &faces,
             const nd_array<Real> &points)
      : _offsets(faces.offsets()), _indices(faces.data()), _points(points) {}

  auto mesh() -> mesh_type {
    return {faces(), tf::make_points<Dims>(_points.make_range()), _cache};
  }

private:
  auto faces() const -> typename mesh_type::faces_type {
    if constexpr (Ngon == 3)
      return tf::make_faces<3>(_indices.make_range());
    else
      return tf::make_faces(_offsets.make_range(), _indices.make_range());
  }

  nd_array<Index> _offsets;
  nd_array<Index> _indices;
  nd_array<Real> _points;
  cpp::cache<Index, Real, Dims, Ngon> _cache;
};

/// @brief The same, for the edge carrier, whose arity is its own.
template <typename Index, typename Real, std::size_t Dims>
class array_edge_mesh {
public:
  using edge_mesh_type = cpp::edge_mesh<Index, Real, Dims>;

  array_edge_mesh(const nd_array<Index> &edges, const nd_array<Real> &points)
      : _edges(edges), _points(points) {}

  auto edge_mesh() -> edge_mesh_type {
    return {tf::make_edges(tf::make_blocked_range<2>(_edges.make_range())),
            tf::make_points<Dims>(_points.make_range()), _cache};
  }

private:
  nd_array<Index> _edges;
  nd_array<Real> _points;
  cpp::edge_mesh_cache<Index, Real, Dims> _cache;
};

} // namespace tf::cpp::detail
