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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./delaunay_execution_tuning.hpp"
#include "./is_owned_constrained_delaunay_face.hpp"
#include <algorithm>
#include <cstddef>
#include <tbb/task_arena.h>

namespace tf::topology::cdt {

template <typename Owner>
auto materialize_constrained_delaunay_faces_parallel(Owner &owner,
                                                     std::size_t n_darts)
    -> bool {
  using Index = typename Owner::index_type;
  if (n_darts < delaunay_execution_tuning::parallel_face_darts ||
      tbb::this_task_arena::max_concurrency() < 2)
    return false;

  const std::size_t chunk_count =
      (n_darts + delaunay_execution_tuning::face_chunk_darts - 1) /
      delaunay_execution_tuning::face_chunk_darts;
  auto &offsets = owner._scratch_face_offsets;
  offsets.allocate_and_initialize(chunk_count + 1, std::size_t(0));
  tf::parallel_for_each(
      tf::make_sequence_range(chunk_count), [&](std::size_t chunk) {
        const std::size_t first =
            chunk * delaunay_execution_tuning::face_chunk_darts;
        const std::size_t last = std::min(
            n_darts, first + delaunay_execution_tuning::face_chunk_darts);
        std::size_t count = 0;
        for (std::size_t edge = first; edge < last; ++edge) {
          Index second = Owner::none;
          Index third = Owner::none;
          if (is_owned_constrained_delaunay_face(
                  owner, static_cast<Index>(edge), second, third))
            ++count;
        }
        offsets[chunk + 1] = count;
      });
  for (std::size_t chunk = 0; chunk < chunk_count; ++chunk)
    offsets[chunk + 1] += offsets[chunk];

  auto &faces = owner._faces;
  auto &first_edges = owner._first_edge_of_triangle;
  if (offsets.back() == 0) {
    faces.clear();
    first_edges.clear();
    return true;
  }
  faces.allocate(offsets.back());
  first_edges.allocate(offsets.back());
  auto *face_data = faces.data_buffer().begin();
  auto *first_data = first_edges.begin();
  tf::parallel_for_each(
      tf::make_sequence_range(chunk_count), [&](std::size_t chunk) {
        const std::size_t first =
            chunk * delaunay_execution_tuning::face_chunk_darts;
        const std::size_t last = std::min(
            n_darts, first + delaunay_execution_tuning::face_chunk_darts);
        auto *vertex_output = face_data + 3 * offsets[chunk];
        auto *first_output = first_data + offsets[chunk];
        for (std::size_t edge = first; edge < last; ++edge) {
          Index second = Owner::none;
          Index third = Owner::none;
          const Index first_edge = static_cast<Index>(edge);
          if (!is_owned_constrained_delaunay_face(owner, first_edge, second,
                                                  third))
            continue;
          *first_output++ = first_edge;
          *vertex_output++ = owner.output_vertex(owner._edges[edge].vertex);
          *vertex_output++ =
              owner.output_vertex(owner._edges[std::size_t(second)].vertex);
          *vertex_output++ =
              owner.output_vertex(owner._edges[std::size_t(third)].vertex);
        }
      });
  return true;
}

} // namespace tf::topology::cdt
