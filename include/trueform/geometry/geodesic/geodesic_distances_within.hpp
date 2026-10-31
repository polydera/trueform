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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/parallel_transform.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/config/verify.hpp"
#include "../../core/polygons.hpp"
#include "../../core/static_size.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../topology/face_membership.hpp"
#include "../../topology/policy/face_membership.hpp"
#include "../../topology/policy/vertex_link.hpp"
#include "../../topology/vertex_link.hpp"
#include "./atomic_values.hpp"
#include "./complete_unfolding_faces.hpp"
#include "./geodesic_front_stats.hpp"
#include "./geodesic_point_state.hpp"
#include "./propagate_geodesic_distances.hpp"
#include "./propagate_geodesic_fronts.hpp"
#include "./unfolding_faces.hpp"
#include "tbb/task_arena.h"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace tf::geometry {

/// The front of @ref tf::geometry::geodesic_distances_within on a tagged
/// mesh, its point state chosen by whether the seeds are several.
template <bool Several, typename RealOut, typename Policy, typename Range,
          typename Stats>
auto geodesic_distances_from_seeds(const tf::polygons<Policy> &polygons,
                                   const Range &seeds, double reach,
                                   Stats &stats) -> tf::buffer<RealOut> {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  using state_t = tf::geometry::geodesic_point_state<Several>;
  constexpr double unreached = std::numeric_limits<double>::infinity();
  const auto points = polygons.points();
  const auto &link = polygons.vertex_link();
  tf::geometry::atomic_values<state_t> reached;
  if constexpr (Several)
    reached.allocate_and_fill(points.size(), unreached,
                              tf::geometry::geodesic_seed_label{
                                  0, std::numeric_limits<float>::infinity()});
  else
    reached.allocate_and_fill(points.size(), unreached);
  std::uint32_t ordinal = 0;
  for (const auto seed : seeds) {
    reached[std::size_t(seed)].distance.store(0.0, std::memory_order_relaxed);
    if constexpr (Several)
      reached[std::size_t(seed)].label.store({ordinal, 0.0f},
                                             std::memory_order_relaxed);
    ++ordinal;
  }

  tf::buffer<tf::geometry::geodesic_front_point<Index>> queue;
  ordinal = 0;
  for (const auto seed : seeds) {
    const std::uint32_t stating = ordinal++;
    const auto source = points[seed].template as<double>();
    for (const auto ring : link[seed]) {
      auto &state = reached[std::size_t(ring)];
      const double held = state.distance.load(std::memory_order_relaxed);
      if (held == 0.0)
        continue;
      const double length =
          (points[ring].template as<double>() - source).length();
      if (length < held) {
        state.distance.store(length, std::memory_order_relaxed);
        if constexpr (Several)
          state.label.store({stating, float(length)},
                            std::memory_order_relaxed);
      }
      std::uint32_t stated = stating;
      if constexpr (Several)
        stated = state.label.load(std::memory_order_relaxed).seed;
      queue.push_back({Index(ring), stated, std::min(held, length)});
    }
  }
  const std::size_t threads =
      std::size_t(tbb::this_task_arena::max_concurrency());
  const std::size_t front_cap =
      threads < 2 ? std::numeric_limits<std::size_t>::max() : 200;
  tf::geometry::unfolding_faces unfolding;
  if constexpr (tf::static_size_v<decltype(polygons.faces()[0])> == 3) {
    unfolding.images.allocate(polygons.size());
    unfolding.formed.allocate(polygons.size());
    if (reach < unreached)
      tf::parallel_fill(unfolding.formed, char(0));
    else
      tf::geometry::complete_unfolding_faces(polygons, unfolding);
  }
  if (!tf::geometry::propagate_geodesic_distances<Several>(
          polygons, unfolding, queue, reached.values(), reach, front_cap,
          stats))
    tf::geometry::propagate_geodesic_fronts<Several>(
        polygons, unfolding, queue, reached.values(), reach, threads, stats);

  tf::buffer<RealOut> distances;
  distances.allocate(points.size());
  tf::parallel_transform(
      reached.values(), distances,
      [reach](const state_t &state) {
        const double d = state.distance.load(std::memory_order_relaxed);
        return d > reach ? std::numeric_limits<RealOut>::infinity()
                         : RealOut(d);
      },
      tf::checked);
  if constexpr (tf::core::verification) {
    tf::buffer<Index> seed_ids;
    for (const auto seed : seeds)
      seed_ids.push_back(Index(seed));
    tf::parallel_for_each(
        tf::make_sequence_range(points.size()),
        [&](std::size_t v) {
          const double d = reached[v].distance.load(std::memory_order_relaxed);
          if (!(d < std::numeric_limits<double>::infinity()))
            return;
          std::uint32_t stated = 0;
          if constexpr (Several)
            stated = reached[v].label.load(std::memory_order_relaxed).seed;
          const double line = (points[v].template as<double>() -
                               points[seed_ids[stated]].template as<double>())
                                  .length();
          tf::core::verify(d >= line * (1.0 - 1e-9),
                           "a geodesic distance is at least the straight "
                           "line to the seed that stated it");
        },
        tf::checked);
  }
  return distances;
}

/// The distances @ref tf::make_geodesic_distances states, within `reach`;
/// an infinite reach is unbounded. The front runs in double precision and
/// the distances are stated in `RealOut` as they are read out.
///
/// An unbounded front reaches every triangle of its pieces, so it forms all
/// their images in one pass first; a front within a finite reach forms each
/// the first time a pop reaches it, and pays for the triangles it reaches.
///
/// `stats` is a @ref tf::geometry::geodesic_front_stats the fronts count
/// into, or `tf::none_t`, which counts nothing.
template <typename RealOut, typename Policy, typename Range, typename Stats>
auto geodesic_distances_within(const tf::polygons<Policy> &polygons,
                               const Range &seeds, double reach, Stats &stats)
    -> tf::buffer<RealOut> {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  if constexpr (!tf::has_face_membership_policy<Policy>) {
    tf::face_membership<Index> membership;
    membership.build(polygons);
    return tf::geometry::geodesic_distances_within<RealOut>(
        polygons | tf::tag(membership), seeds, reach, stats);
  } else if constexpr (!tf::has_vertex_link_policy<Policy>) {
    tf::vertex_link<Index> link;
    link.build(polygons.faces(), polygons.face_membership());
    return tf::geometry::geodesic_distances_within<RealOut>(
        polygons | tf::tag(link), seeds, reach, stats);
  } else {
    std::size_t n_seeds = 0;
    for (const auto seed : seeds) {
      static_cast<void>(seed);
      ++n_seeds;
    }
    if (n_seeds > 1)
      return tf::geometry::geodesic_distances_from_seeds<true, RealOut>(
          polygons, seeds, reach, stats);
    return tf::geometry::geodesic_distances_from_seeds<false, RealOut>(
        polygons, seeds, reach, stats);
  }
}

} // namespace tf::geometry
