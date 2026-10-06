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
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/buffer.hpp"
#include "../core/checked.hpp"
#include "../core/none.hpp"
#include "../core/polygons.hpp"
#include "../core/polygons_buffer.hpp"
#include "../core/range.hpp"
#include "../core/static_size.hpp"
#include "../core/views/drop.hpp"
#include "../core/views/sequence_range.hpp"
#include "../core/views/take.hpp"
#include "./hole_fill_result.hpp"
#include <cstddef>
#include <type_traits>

namespace tf {

/// @ingroup fill
/// @brief The mesh a fill states: the input, with what it changed.
///
/// A face no split reached is copied at its own arity; a carrying face a
/// split reached is replaced by its plan, the one authority on what that
/// face became; and the patches follow. The points are the input's own and
/// then the minted lane read flat, which is the identity space the result's
/// corners already speak.
///
/// The output keeps the input's arity where the input has one: an
/// all-triangle mesh comes back all-triangle, anything else comes back
/// variable, since plans and patches are triangles whatever the faces
/// around them are.
///
/// @tparam Index The index type the faces are written in; defaults to the
///   input's own.
/// @tparam Policy The polygons policy type.
/// @tparam RimIndex The index type the result is stated in.
/// @tparam RealT The coordinate type.
/// @param polygons The mesh the fill was stated against.
/// @param result The @ref tf::hole_fill_result to replay.
/// @return The filled mesh.
template <typename Index = tf::none_t, typename Policy, typename RimIndex,
          typename RealT>
auto make_filled_mesh(const tf::polygons<Policy> &polygons,
                      const tf::hole_fill_result<RimIndex, RealT> &result) {
  using OutIndex =
      std::conditional_t<std::is_same_v<Index, tf::none_t>,
                         std::decay_t<decltype(polygons.faces()[0][0])>, Index>;
  constexpr std::size_t ngon =
      tf::static_size_v<std::decay_t<decltype(polygons.faces()[0])>>;
  constexpr std::size_t out_ngon = ngon == 3 ? 3 : tf::dynamic_size;

  const auto faces = polygons.faces();
  const auto points = polygons.points();
  const auto n_faces = faces.size();
  const auto n_points = points.size();

  tf::buffer<OutIndex> plan_of;
  plan_of.allocate(n_faces);
  tf::parallel_fill(plan_of, OutIndex(-1));
  tf::parallel_for_each(tf::make_sequence_range(result.plan_faces.size()),
                        [&plan_of, &result](std::size_t plan) {
                          plan_of[std::size_t(result.plan_faces[plan])] =
                              OutIndex(plan);
                        },
                        tf::checked);

  tf::buffer<OutIndex> face_offsets;
  face_offsets.allocate(n_faces + 1);
  face_offsets[0] = 0;
  for (std::size_t face = 0; face < n_faces; ++face) {
    const auto plan = plan_of[face];
    face_offsets[face + 1] =
        OutIndex(face_offsets[face] +
                 OutIndex(plan < 0 ? 1 : result.plan_triangles[std::size_t(
                                                                   plan)]
                                             .size()));
  }
  const auto n_patch = result.triangles.data_buffer().size();
  const auto n_out = std::size_t(face_offsets[n_faces]) + n_patch;

  tf::polygons_buffer<OutIndex, RealT, 3, out_ngon> out;
  out.points_buffer().allocate(n_points +
                               result.minted_points.data_buffer().size());
  tf::parallel_copy(points, tf::take(out.points(), n_points));
  tf::parallel_copy(tf::make_range(result.minted_points.data_buffer()),
                    tf::drop(out.points(), n_points));

  if constexpr (out_ngon == 3) {
    out.faces_buffer().allocate(n_out);
    auto blocks = out.faces();
    tf::parallel_for_each(
        tf::make_sequence_range(n_faces),
        [&blocks, &faces, &plan_of, &face_offsets, &result](std::size_t face) {
          const auto plan = plan_of[face];
          auto slot = std::size_t(face_offsets[face]);
          if (plan < 0) {
            const auto corners = faces[face];
            for (std::size_t k = 0; k < 3; ++k)
              blocks[slot][k] = OutIndex(corners[k]);
            return;
          }
          const auto triangles = result.plan_triangles[std::size_t(plan)];
          for (const auto &triangle : triangles) {
            for (std::size_t k = 0; k < 3; ++k)
              blocks[slot][k] = OutIndex(triangle[k]);
            ++slot;
          }
        });
    const auto base = std::size_t(face_offsets[n_faces]);
    tf::parallel_for_each(
        tf::make_sequence_range(n_patch),
        [&blocks, &result, base](std::size_t patch) {
          const auto &triangle = result.triangles.data_buffer()[patch];
          for (std::size_t k = 0; k < 3; ++k)
            blocks[base + patch][k] = OutIndex(triangle[k]);
        });
  } else {
    auto &offsets = out.faces_buffer().offsets_buffer();
    auto &corners = out.faces_buffer().data_buffer();
    offsets.allocate(n_out + 1);
    offsets[0] = 0;
    std::size_t slot = 0;
    for (std::size_t face = 0; face < n_faces; ++face) {
      const auto plan = plan_of[face];
      if (plan < 0) {
        offsets[slot + 1] =
            OutIndex(offsets[slot] + OutIndex(faces[face].size()));
        ++slot;
        continue;
      }
      for (std::size_t k = 0;
           k < result.plan_triangles[std::size_t(plan)].size(); ++k) {
        offsets[slot + 1] = OutIndex(offsets[slot] + 3);
        ++slot;
      }
    }
    for (std::size_t patch = 0; patch < n_patch; ++patch) {
      offsets[slot + 1] = OutIndex(offsets[slot] + 3);
      ++slot;
    }
    corners.allocate(std::size_t(offsets[n_out]));
    auto blocks = out.faces();
    tf::parallel_for_each(
        tf::make_sequence_range(n_faces),
        [&blocks, &faces, &plan_of, &face_offsets, &result](std::size_t face) {
          const auto plan = plan_of[face];
          auto at = std::size_t(face_offsets[face]);
          if (plan < 0) {
            const auto source = faces[face];
            auto block = blocks[at];
            for (std::size_t k = 0; k < source.size(); ++k)
              block[k] = OutIndex(source[k]);
            return;
          }
          const auto triangles = result.plan_triangles[std::size_t(plan)];
          for (const auto &triangle : triangles) {
            auto block = blocks[at++];
            for (std::size_t k = 0; k < 3; ++k)
              block[k] = OutIndex(triangle[k]);
          }
        });
    const auto base = std::size_t(face_offsets[n_faces]);
    tf::parallel_for_each(tf::make_sequence_range(n_patch),
                          [&blocks, &result, base](std::size_t patch) {
                            const auto &triangle =
                                result.triangles.data_buffer()[patch];
                            auto block = blocks[base + patch];
                            for (std::size_t k = 0; k < 3; ++k)
                              block[k] = OutIndex(triangle[k]);
                          });
  }
  return out;
}

} // namespace tf
