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

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../topology/topo_id.hpp"
#include <array>
#include <cassert>
#include <cstddef>
#include <utility>

namespace tf::arrangement {

/// Remove triangles collapsed by a closed weld after their edge definitions
/// have already entered the fused PA generation.
///
/// The final triangle store is plane- and member-ordered, so one keep prefix is
/// also the exact remap for plane ranges, face ranges, and coplanar survivor
/// tickets. Every per-triangle table rides that one prefix, and coplanar
/// descriptors are compacted with their owning duplicate.
template <typename Index, typename Coplanar>
auto compact_plane_weld_triangles(
    tf::buffer<std::array<Index, 3>> &triangles,
    tf::buffer<std::array<Index, 3>> &slots,
    tf::buffer<std::array<tf::topo_id<short>, 3>> &subs,
    tf::buffer<Index> &coplanar_of, tf::buffer<Coplanar> &coplanar,
    tf::buffer<char> &stacked, tf::buffer<Index> &cells,
    tf::buffer<Index> &plane_offsets,
    tf::buffer<std::array<Index, 2>> &face_ranges) -> std::size_t {
  const auto n_triangles = triangles.size();
  if (n_triangles == 0)
    return 0;
  // the arrangement facts a triangle carries are a requested product; unasked,
  // there is no row and the survivor a duplicate would name does not exist
  const bool with_arrangement = coplanar_of.size() == n_triangles;
  const auto coplanar_link = [&](std::size_t triangle) {
    return with_arrangement ? coplanar_of[triangle] : Index(-1);
  };

  tf::buffer<char> nondegenerate;
  nondegenerate.allocate(n_triangles);
  tf::parallel_for_each(
      tf::make_sequence_range(n_triangles),
      [&](std::size_t triangle) {
        const auto &corners = triangles[triangle];
        nondegenerate[triangle] =
            char(corners[0] != corners[1] && corners[1] != corners[2] &&
                 corners[0] != corners[2]);
      },
      tf::checked);

  tf::buffer<Index> triangle_prefix;
  triangle_prefix.allocate(n_triangles + 1);
  triangle_prefix[0] = 0;
  tf::parallel_for_each(
      tf::make_sequence_range(n_triangles),
      [&](std::size_t triangle) {
        bool keep = nondegenerate[triangle] != 0;
        const auto descriptor = coplanar_link(triangle);
        if (keep && descriptor != Index(-1)) {
          const auto survivor = coplanar[std::size_t(descriptor)].survivor;
          assert(nondegenerate[triangle] ==
                 nondegenerate[std::size_t(survivor)]);
          keep = nondegenerate[std::size_t(survivor)] != 0;
        }
        triangle_prefix[triangle + 1] = keep ? Index(1) : Index(0);
      },
      tf::checked);
  for (std::size_t triangle = 1; triangle < triangle_prefix.size(); ++triangle)
    triangle_prefix[triangle] += triangle_prefix[triangle - 1];
  const auto n_final = std::size_t(triangle_prefix[n_triangles]);
  if (n_final == n_triangles)
    return 0;

  tf::buffer<Index> dead_prefix;
  dead_prefix.allocate(n_triangles + 1);
  dead_prefix[0] = 0;
  tf::parallel_for_each(
      tf::make_sequence_range(n_triangles),
      [&](std::size_t triangle) {
        dead_prefix[triangle + 1] =
            triangle_prefix[triangle] != triangle_prefix[triangle + 1] &&
                    coplanar_link(triangle) != Index(-1)
                ? Index(1)
                : Index(0);
      },
      tf::checked);
  for (std::size_t triangle = 1; triangle < dead_prefix.size(); ++triangle)
    dead_prefix[triangle] += dead_prefix[triangle - 1];

  tf::buffer<std::array<Index, 3>> final_triangles;
  tf::buffer<std::array<Index, 3>> final_slots;
  tf::buffer<std::array<tf::topo_id<short>, 3>> final_subs;
  tf::buffer<Index> final_coplanar_of;
  tf::buffer<char> final_stacked;
  final_triangles.allocate(n_final);
  final_subs.allocate(n_final);
  tf::buffer<Coplanar> final_coplanar;
  if (with_arrangement) {
    final_slots.allocate(n_final);
    final_coplanar_of.allocate(n_final);
    final_stacked.allocate(n_final);
    final_coplanar.allocate(std::size_t(dead_prefix[n_triangles]));
  }
  // recorded cells ride the same mask: rows go, the cells they name stay
  const bool with_cells = cells.size() == n_triangles;
  tf::buffer<Index> final_cells;
  if (with_cells)
    final_cells.allocate(n_final);
  tf::parallel_for_each(
      tf::make_sequence_range(n_triangles),
      [&](std::size_t triangle) {
        if (triangle_prefix[triangle] == triangle_prefix[triangle + 1])
          return;
        const auto out = std::size_t(triangle_prefix[triangle]);
        final_triangles[out] = triangles[triangle];
        final_subs[out] = subs[triangle];
        if (with_cells)
          final_cells[out] = cells[triangle];
        if (!with_arrangement)
          return;
        final_slots[out] = slots[triangle];
        final_stacked[out] = stacked[triangle];
        const auto descriptor = coplanar_of[triangle];
        if (descriptor == Index(-1)) {
          final_coplanar_of[out] = Index(-1);
          return;
        }
        const auto out_descriptor = dead_prefix[triangle];
        final_coplanar_of[out] = out_descriptor;
        auto value = coplanar[std::size_t(descriptor)];
        assert(triangle_prefix[std::size_t(value.survivor)] !=
               triangle_prefix[std::size_t(value.survivor) + 1]);
        value.survivor = triangle_prefix[std::size_t(value.survivor)];
        final_coplanar[std::size_t(out_descriptor)] = value;
      },
      tf::checked);

  tf::parallel_for_each(
      tf::make_sequence_range(plane_offsets.size()),
      [&](std::size_t plane) {
        plane_offsets[plane] =
            triangle_prefix[std::size_t(plane_offsets[plane])];
      },
      tf::checked);
  tf::parallel_for_each(
      tf::make_sequence_range(face_ranges.size()),
      [&](std::size_t face) {
        const auto range = face_ranges[face];
        face_ranges[face] = {triangle_prefix[std::size_t(range[0])],
                             triangle_prefix[std::size_t(range[1])]};
      },
      tf::checked);

  triangles = std::move(final_triangles);
  subs = std::move(final_subs);
  if (with_arrangement) {
    slots = std::move(final_slots);
    coplanar_of = std::move(final_coplanar_of);
    coplanar = std::move(final_coplanar);
    stacked = std::move(final_stacked);
  }
  if (with_cells)
    cells = std::move(final_cells);
  else
    cells.clear();
  return n_triangles - n_final;
}

} // namespace tf::arrangement
