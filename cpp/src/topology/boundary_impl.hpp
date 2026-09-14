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

#include "trueform/cpp/topology/boundary_curves.hpp"
#include "trueform/cpp/topology/boundary_edges.hpp"
#include "trueform/cpp/topology/boundary_paths.hpp"

#include "trueform/core/algorithm/parallel_copy.hpp"
#include "trueform/core/algorithm/parallel_transform.hpp"
#include "trueform/core/points.hpp"
#include "trueform/core/range.hpp"
#include "trueform/core/views/indirect_range.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/topology.hpp"

#include <tbb/parallel_sort.h>

#include <algorithm>
#include <cstddef>
#include <utility>

namespace tf::cpp {
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto boundary_edges(const mesh<Index, Real, Dims, Ngon> &value)
    -> nd_array<Index> {
  auto edges = tf::make_boundary_edges(value.faces(), value.face_membership());
  const auto count = static_cast<int>(edges.size());
  return nd_array<Index>::from_buffer(std::move(edges.data_buffer()),
                                      {count, 2});
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto boundary_paths(const mesh<Index, Real, Dims, Ngon> &value)
    -> offset_blocked_buffer<Index, Index> {
  auto paths = tf::make_boundary_paths(value.faces(), value.face_membership());
  return offset_blocked_buffer<Index, Index>::from_buffer(std::move(paths));
}

/// The paths name the mesh's own vertices, so the points they reach are
/// gathered once and the paths are rewritten to that dense numbering.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto boundary_curves(const mesh<Index, Real, Dims, Ngon> &value)
    -> boundary_curves_result<Index, Real, Dims> {
  auto paths = tf::make_boundary_paths(value.faces(), value.face_membership());
  tf::buffer<Index> unique_ids;
  unique_ids.allocate(paths.data_buffer().size());
  std::copy(paths.data_buffer().begin(), paths.data_buffer().end(),
            unique_ids.begin());
  tbb::parallel_sort(unique_ids.begin(), unique_ids.end());
  unique_ids.erase_till_end(std::unique(unique_ids.begin(), unique_ids.end()));

  tf::parallel_transform(
      paths.data_buffer(), paths.data_buffer(),
      [&unique_ids](Index id) {
        return static_cast<Index>(
            std::lower_bound(unique_ids.begin(), unique_ids.end(), id) -
            unique_ids.begin());
      },
      tf::checked);

  tf::buffer<Real> point_data;
  point_data.allocate(unique_ids.size() * Dims);
  tf::parallel_copy(
      tf::make_indirect_range(unique_ids, value.points()),
      tf::make_points<Dims>(
          tf::make_range(point_data.begin(), point_data.end())));

  return {offset_blocked_buffer<Index, Index>::from_buffer(std::move(paths)),
          nd_array<Real>::from_buffer(
              std::move(point_data),
              {static_cast<int>(unique_ids.size()), static_cast<int>(Dims)})};
}

} // namespace tf::cpp
