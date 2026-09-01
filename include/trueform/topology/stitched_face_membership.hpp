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
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/algorithm/reduce.hpp"
#include "../core/buffer.hpp"
#include "../core/checked.hpp"
#include "../core/faces.hpp"
#include "../core/static_size.hpp"
#include "../core/stitch_index_map.hpp"
#include "../core/views/indirect_range.hpp"
#include "../core/views/mapped_range.hpp"
#include "../core/views/sequence_range.hpp"
#include "./face_membership.hpp"
#include "./face_membership_like.hpp"
#include <algorithm>
#include <cstddef>
#include <functional>

namespace tf {

/// @ingroup topology_connectivity
/// @brief Face membership of a merged mesh, reusing the sources'.
///
/// A face that survived whole is the same polygon it was, so each of its
/// vertices keeps that incidence and only the face id is remapped through
/// `sim.face_f`. Incidences on new faces have no counterpart and are built
/// from the result geometry.
///
/// @tparam Index The integer type for indices.
/// @param result_faces The faces of the merged mesh.
/// @param fm0 Face membership of the first source mesh.
/// @param fm1 Face membership of the second source mesh.
/// @param sim The merge's @ref tf::stitch_index_map (two sources).
/// @return The merged mesh's face membership.
template <typename Index, typename FacesPolicy, typename FMPolicy0,
          typename FMPolicy1>
auto stitched_face_membership(const tf::faces<FacesPolicy> &result_faces,
                              const tf::face_membership_like<FMPolicy0> &fm0,
                              const tf::face_membership_like<FMPolicy1> &fm1,
                              const tf::stitch_index_map<Index> &sim)
    -> tf::face_membership<Index> {
  const Index n_points = sim.n_output_points;
  const Index gone = sim.n_output_faces;

  auto with_fm = [&](Index source, auto &&fn) {
    if (source == 0)
      fn(fm0);
    else
      fn(fm1);
  };

  auto new_faces = tf::make_indirect_range(sim.new_faces, result_faces);
  const Index total_new_size = [&] {
    if constexpr (tf::static_size_v<decltype(result_faces[0])> !=
                  tf::dynamic_size)
      return Index(sim.new_faces.size()) *
             Index(tf::static_size_v<decltype(result_faces[0])>);
    else
      return tf::reduce(
          tf::make_mapped_range(new_faces,
                                [](const auto &f) -> Index { return f.size(); }),
          std::plus<>(), Index(0), tf::checked);
  }();

  tf::face_membership<Index> new_fm;
  new_fm.build(tf::make_faces(new_faces), std::size_t(n_points),
               total_new_size);

  tf::buffer<Index> offsets;
  offsets.allocate(static_cast<std::size_t>(n_points) + 1);
  tf::parallel_fill(offsets, 0);

  tf::parallel_for_each(
      tf::make_sequence_range(Index(0), n_points),
      [&](Index o) {
        Index count = 0;
        const Index source = sim.point_sources[o];
        if (source != sim.n_sources)
          with_fm(source, [&](const auto &fm) {
            auto forward = sim.face_f[source];
            for (auto face_id : fm[sim.point_ids[o]])
              if (forward[face_id] != gone)
                ++count;
          });
        offsets[o + 1] = count + Index(new_fm[o].size());
      },
      tf::checked);

  for (Index o = 0; o < n_points; ++o)
    offsets[o + 1] += offsets[o];

  tf::face_membership<Index> result;
  result.data_buffer().allocate(offsets.back());
  auto &data = result.data_buffer();

  tf::parallel_for_each(
      tf::make_sequence_range(Index(0), n_points),
      [&](Index o) {
        auto dest = data.begin() + offsets[o];
        const Index source = sim.point_sources[o];
        if (source != sim.n_sources)
          with_fm(source, [&](const auto &fm) {
            auto forward = sim.face_f[source];
            for (auto face_id : fm[sim.point_ids[o]])
              if (forward[face_id] != gone)
                *dest++ = forward[face_id];
          });
        for (auto local_id : new_fm[o])
          *dest++ = sim.new_faces[local_id];
        std::sort(data.begin() + offsets[o], data.begin() + offsets[o + 1],
                  std::greater<Index>());
      },
      tf::checked);

  result.offsets_buffer() = std::move(offsets);

  return result;
}

} // namespace tf
