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

#include "trueform/core/algorithm/parallel_copy.hpp"
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/mesh.hpp"

#include <cstddef>

namespace tf::cpp {
namespace detail {

/// A mesh's faces in core's own storage. An entry that answers with a mesh
/// answers with `tf::polygons_buffer`, and a reading is BORROWED, so what the
/// caller is handed stands on arrays of its own.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto materialize_faces(const mesh<Index, Real, Dims, Ngon> &value,
                       tf::polygons_buffer<Index, Real, Dims, Ngon> &output)
    -> void {
  const auto &geometry = value.geometry();
  auto &faces = output.faces_buffer();
  if constexpr (Ngon == 3) {
    faces.allocate(value.number_of_faces());
  } else {
    faces.offsets_buffer().allocate(geometry.offsets.size());
    tf::parallel_copy(geometry.offsets, faces.offsets_buffer());
    faces.data_buffer().allocate(geometry.indices.size());
  }
  tf::parallel_copy(geometry.indices, faces.data_buffer());
}

/// The same reading whole, at the arity it was read in.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto materialized(const mesh<Index, Real, Dims, Ngon> &value)
    -> tf::polygons_buffer<Index, Real, Dims, Ngon> {
  tf::polygons_buffer<Index, Real, Dims, Ngon> output;
  detail::materialize_faces(value, output);
  output.points_buffer().allocate(value.number_of_points());
  tf::parallel_copy(value.geometry().coordinates,
                    output.points_buffer().data_buffer());
  return output;
}

} // namespace detail
} // namespace tf::cpp
