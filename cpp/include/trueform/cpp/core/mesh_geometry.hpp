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

#include "trueform/core/faces.hpp"
#include "trueform/core/points.hpp"
#include "trueform/core/range.hpp"
#include "trueform/core/static_size.hpp"
#include "trueform/cpp/core/detail/mesh_faces_view.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace tf::cpp {

/// @brief One reading of a mesh: the arrays it stands on, and the two stamps
/// that name it.
///
/// The arrays are borrowed — whoever holds this reading answers for them — and
/// the arity is in the type, so the faces are a fixed block of three or the
/// blocks the offsets state, and never a runtime question.
///
/// A caller states which reading it wants answered for. The faces stamp moves
/// when the faces are restated; the points stamp moves when any point does. A
/// structure that also depends on how MANY points there are reads that from the
/// points themselves — the count is a property of the geometry, not a third
/// stamp.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
struct mesh_geometry {
  using index_range = tf::range<const Index *, tf::dynamic_size>;
  using real_range = tf::range<const Real *, tf::dynamic_size>;
  using faces_type = typename detail::mesh_faces_view<Index, Ngon>::type;
  using points_type = decltype(tf::make_points<Dims>(
      std::declval<tf::range<const Real *, tf::dynamic_size>>()));

  /// The offsets a mixed mesh states its faces with, and what reads any
  /// per-face-side array back at that arity. A triangle mesh states three
  /// consecutive sides and carries none.
  index_range offsets;
  index_range indices;
  real_range coordinates;
  std::uint64_t faces_stamp = 0;
  std::uint64_t points_stamp = 0;

  auto faces() const -> faces_type {
    if constexpr (Ngon == 3)
      return tf::make_faces<3>(indices);
    else
      return tf::make_faces(offsets, indices);
  }

  auto points() const -> points_type {
    return tf::make_points<Dims>(coordinates);
  }

  auto number_of_faces() const -> std::size_t {
    if constexpr (Ngon == 3)
      return indices.size() / 3;
    else
      return offsets.size() ? offsets.size() - 1 : 0;
  }

  auto number_of_points() const -> std::size_t {
    return coordinates.size() / Dims;
  }
};

} // namespace tf::cpp
