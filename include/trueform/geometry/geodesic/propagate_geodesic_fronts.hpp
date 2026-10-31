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
#include "../../core/buffer.hpp"
#include "../../core/none.hpp"
#include "../../core/polygons.hpp"
#include "../../core/range.hpp"
#include "../../core/static_size.hpp"
#include "./complete_unfolding_faces.hpp"
#include "./geodesic_front_stats.hpp"
#include "./geodesic_point_state.hpp"
#include "./propagate_geodesic_distances.hpp"
#include "./unfolding_faces.hpp"
#include "tbb/task_group.h"
#include <algorithm>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace tf::geometry {

/// Run a front that outgrew one heap as `n_fronts` heaps at once.
///
/// The front is ordered by point id and cut into at most `n_fronts`
/// contiguous slices of equal length, the last taking the rest, and each runs
/// through @ref tf::geometry::propagate_geodesic_distances on its own heap
/// until it settles. Mesh order keeps a slice's points together, so each
/// front grows its own stretch of the wave. The heaps share `distances`: a
/// point one front lowered is read by every other, and a front stops where
/// another already stands nearer. The images every front reads are formed
/// first, so the fronts only read them. Each front counts its pops on its
/// own, and they enter `stats` after the fronts join.
template <bool Several, typename Policy, typename Index, typename Stats>
auto propagate_geodesic_fronts(
    const tf::polygons<Policy> &polygons,
    tf::geometry::unfolding_faces &unfolding,
    tf::buffer<tf::geometry::geodesic_front_point<Index>> &queue,
    tf::range<tf::geometry::geodesic_point_state<Several> *, tf::dynamic_size>
        states,
    double reach, std::size_t n_fronts, Stats &stats) -> void {
  std::sort(queue.begin(), queue.end(),
            [](const tf::geometry::geodesic_front_point<Index> &x,
               const tf::geometry::geodesic_front_point<Index> &y) {
              return x.id < y.id;
            });
  if constexpr (tf::static_size_v<decltype(polygons.faces()[0])> == 3)
    tf::geometry::complete_unfolding_faces(polygons, unfolding);
  const std::size_t size = queue.size();
  const std::size_t step =
      std::max(std::size_t(1), (size + n_fronts - 1) / n_fronts);
  tf::buffer<tf::geometry::geodesic_front_stats> counted;
  counted.allocate((size + step - 1) / step);
  std::fill(counted.begin(), counted.end(),
            tf::geometry::geodesic_front_stats{});
  tbb::task_group fronts;
  for (std::size_t from = 0; from < size; from += step)
    fronts.run([&, from] {
      tf::buffer<tf::geometry::geodesic_front_point<Index>> front;
      front.allocate(std::min(step, size - from));
      std::copy(queue.begin() + from, queue.begin() + from + front.size(),
                front.begin());
      if constexpr (std::is_same_v<std::decay_t<Stats>, tf::none_t>)
        tf::geometry::propagate_geodesic_distances<Several>(
            polygons, unfolding, front, states, reach,
            std::numeric_limits<std::size_t>::max(), stats);
      else
        tf::geometry::propagate_geodesic_distances<Several>(
            polygons, unfolding, front, states, reach,
            std::numeric_limits<std::size_t>::max(), counted[from / step]);
    });
  fronts.wait();
  for (const auto &front : counted)
    tf::geometry::record_geodesic_front(stats, front.pops);
}

} // namespace tf::geometry
