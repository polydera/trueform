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
#include "./is_owned_unconstrained_delaunay_face.hpp"
#include "./materialize_unconstrained_delaunay_faces_serial.hpp"
#include <algorithm>
#include <cstddef>
#include <tbb/task_arena.h>

namespace tf::topology::cdt {

template <typename Owner, typename VisitBuffer>
auto materialize_unconstrained_delaunay_faces_parallel(Owner &owner,
                                                       VisitBuffer &visited)
    -> void {
  using Index = typename Owner::index_type;
  using VertexPolicy = typename Owner::vertex_policy;
  if (owner._edges.size() < delaunay_execution_tuning::parallel_face_darts ||
      tbb::this_task_arena::max_concurrency() < 2) {
    materialize_unconstrained_delaunay_faces_serial(owner, visited);
    return;
  }

  const std::size_t chunk_count =
      (owner._edges.size() + delaunay_execution_tuning::face_chunk_darts - 1) /
      delaunay_execution_tuning::face_chunk_darts;
  owner._face_offsets.allocate_and_initialize(chunk_count + 1, std::size_t(0));
  tf::parallel_for_each(
      tf::make_sequence_range(chunk_count), [&](std::size_t chunk) {
        const std::size_t first_dart =
            chunk * delaunay_execution_tuning::face_chunk_darts;
        const std::size_t last_dart =
            std::min(owner._edges.size(),
                     first_dart + delaunay_execution_tuning::face_chunk_darts);
        std::size_t count = 0;
        for (std::size_t dart = first_dart; dart < last_dart; ++dart) {
          if (owner._edges[dart].vertex == Owner::none)
            continue;
          const auto orbit = trace_unconstrained_delaunay_triangle(
              owner, static_cast<Index>(dart));
          if (is_owned_unconstrained_delaunay_face(owner, visited, orbit))
            ++count;
        }
        owner._face_offsets[chunk + 1] = count;
      });
  for (std::size_t chunk = 0; chunk < chunk_count; ++chunk)
    owner._face_offsets[chunk + 1] += owner._face_offsets[chunk];

  owner._faces.allocate(owner._face_offsets.back());
  auto *face_data = owner._faces.data_buffer().begin();
  tf::parallel_for_each(
      tf::make_sequence_range(chunk_count), [&](std::size_t chunk) {
        const std::size_t first_dart =
            chunk * delaunay_execution_tuning::face_chunk_darts;
        const std::size_t last_dart =
            std::min(owner._edges.size(),
                     first_dart + delaunay_execution_tuning::face_chunk_darts);
        auto *output = face_data + 3 * owner._face_offsets[chunk];
        for (std::size_t dart = first_dart; dart < last_dart; ++dart) {
          if (owner._edges[dart].vertex == Owner::none)
            continue;
          const Index first = static_cast<Index>(dart);
          const auto orbit =
              trace_unconstrained_delaunay_triangle(owner, first);
          if (!is_owned_unconstrained_delaunay_face(owner, visited, orbit))
            continue;
          const Index second = orbit.darts[1];
          const Index third = orbit.darts[2];
          const Index first_vertex = owner._edges[dart].vertex;
          const Index second_vertex = owner._edges[std::size_t(second)].vertex;
          const Index third_vertex = owner._edges[std::size_t(third)].vertex;
          *output++ = VertexPolicy::output(
              first_vertex, owner._sites[std::size_t(first_vertex)]);
          *output++ = VertexPolicy::output(
              second_vertex, owner._sites[std::size_t(second_vertex)]);
          *output++ = VertexPolicy::output(
              third_vertex, owner._sites[std::size_t(third_vertex)]);
        }
      });
}

} // namespace tf::topology::cdt
