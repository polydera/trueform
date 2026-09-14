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
#include "../../core/point.hpp"
#include "../../core/range.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./gather_vertex_candidates.hpp"
#include "./place_vertex.hpp"
#include "./placement_tables.hpp"
#include "./pool/certificate_census.hpp"
#include "./pool/certificate_lane.hpp"
#include "./pool/certify_exact.hpp"
#include "./pool/exact_lane.hpp"
#include "./pool/pool_input_planes.hpp"
#include "./quantized_plane.hpp"

#include <cstddef>

namespace tf::exact::door {

/// The cascade alone, for a lattice whose width states no certificate.
template <typename Index, typename Int, typename RealType>
auto place_cascade_vertices(const placement_tables<Index, Int, RealType> &tables,
                            Int tolerance,
                            tf::buffer<tf::point<Int, 3>> &placed) -> void {
  struct local_t {
    tf::small_vector<quantized_plane<Int>, 16> candidates;
  };
  placed.allocate(tables.points.size());
  tf::parallel_for_each(
      tf::make_sequence_range(Index(tables.points.size())),
      [&tables, &placed, tolerance](Index flat, local_t &local) {
        const auto direction =
            gather_vertex_candidates(tables, flat, tolerance, local.candidates);
        placed[std::size_t(flat)] =
            place_vertex(tables.points[std::size_t(flat)],
                         tf::make_range(local.candidates.begin(),
                                        local.candidates.end()),
                         direction, tolerance)
                .point;
      },
      local_t{}, tf::checked(64));
}

/// The same traversal with RANK 1 READING THE PARTITION. A vertex whose own
/// target name committed to a pooled exact plane is placed on THAT plane, so
/// two walls the partition joined arrive lattice-exactly coplanar; every
/// other vertex takes the cascade unchanged, and a feature vertex is never
/// in the table at all.
///
/// The certificate cannot refuse what the partition admitted — a name is
/// certified on its whole support and this vertex is in it — so the fall
/// through to the cascade is an invariant guard and not a second policy.
///
/// The lane holds the frames of the LAST pool this block asked about. A
/// wall's vertices are consecutive in its own mesh, so one reduction answers
/// a whole wall, and a miss costs one frame and never a wrong answer.
///
/// The certificate's own tally is block scratch and not a product: the
/// verdict this pass reads was the partition's, and the partition's census is
/// what the door publishes.
template <typename Index, typename Int, typename RealType>
auto place_pooled_vertices(const placement_tables<Index, Int, RealType> &tables,
                           const pool::input_pools<Int> &pools, Int tolerance,
                           tf::buffer<tf::point<Int, 3>> &placed) -> void {
  struct local_t {
    tf::small_vector<quantized_plane<Int>, 16> candidates;
    pool::certificate_lane<Int> lane{};
    pool::certificate_census census{};
    int held = -1;
  };
  placed.allocate(tables.points.size());
  const bool committed = pools.pool_of_vertex.size() == tables.points.size();
  tf::parallel_for_each(
      tf::make_sequence_range(Index(tables.points.size())),
      [&tables, &pools, &placed, tolerance, committed](Index flat,
                                                       local_t &local) {
        const auto &original = tables.points[std::size_t(flat)];
        const int pool =
            committed ? pools.pool_of_vertex[std::size_t(flat)] : -1;
        if (pool >= 0) {
          if (local.held != pool) {
            local.lane = pool::certificate_lane<Int>{};
            local.held = pool;
          }
          pool::solved_step<Int> step{};
          const auto certificate = pool::certify_exact<Int>(
              pools.plane[std::size_t(pool)], original, tolerance, local.lane,
              step, local.census);
          if (certificate.certified) {
            placed[std::size_t(flat)] = certificate.point;
            return;
          }
        }
        const auto direction =
            gather_vertex_candidates(tables, flat, tolerance, local.candidates);
        placed[std::size_t(flat)] =
            place_vertex(original,
                         tf::make_range(local.candidates.begin(),
                                        local.candidates.end()),
                         direction, tolerance)
                .point;
      },
      local_t{}, tf::checked(64));
}

/// Every original vertex of every form placed on the lattice in one
/// traversal of the flat vertex space — the largest independent carrier
/// the question has, since a placement reads only the names its own
/// incident faces stated.
///
/// The product is the placed table by flat id. Nothing about a tag enters
/// here; the tables carry the offsets that made the space flat.
///
/// The cutoff is the vertex count at which the threading pays: a placement
/// searches the widest triple of its candidate list, so the per-vertex cost
/// is cubic in a valence and not a constant.
///
/// A lattice with no arithmetic lane has no partition to read, so it takes
/// the cascade alone.
template <typename Index, typename Int, typename RealType>
auto place_vertices(const placement_tables<Index, Int, RealType> &tables,
                    const pool::input_pools<Int> &pools, Int tolerance,
                    tf::buffer<tf::point<Int, 3>> &placed) -> void {
  if constexpr (pool::has_exact_lane<Int>)
    place_pooled_vertices(tables, pools, tolerance, placed);
  else
    place_cascade_vertices(tables, tolerance, placed);
}

} // namespace tf::exact::door
