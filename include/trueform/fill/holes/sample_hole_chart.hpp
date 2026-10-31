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
#include "../../core/blocked_buffer.hpp"
#include "../../core/buffer.hpp"
#include "../../core/edges.hpp"
#include "../../core/point.hpp"
#include "../../core/points.hpp"
#include "../../core/views/take.hpp"
#include "../../topology/cdt/constrained_delaunay_refinement_owner.hpp"
#include "../../topology/cdt_refine_config.hpp"
#include "../../topology/cdt_refine_status.hpp"
#include "../../topology/cdt_refiner.hpp"
#include "../../topology/cdt_region_mode.hpp"
#include "../hole_fill_status.hpp"
#include "../hole_split.hpp"
#include "./hole_chart.hpp"
#include "./hole_collar.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::fill {

/// The quality floor the chart is refined to, in @ref tf::triangle_quality
/// units: sample spacing starts at the local rim edge and grows away from the
/// rim at the rate this floor allows. The seam a chart patch states is the
/// secant of the surface across its first ring, which this floor sets: on a
/// 72-vertex rim of a 50-degree spherical cap it reads about 8.4 degrees, and
/// it halves as the rim's edges halve.
inline constexpr float hole_chart_sample_quality = 0.375f;

/// One split the sampling made of a rim edge: the edge, leaving rim position
/// `edge`, and the split's parameter along it from that position, a
/// numerator over @ref tf::hole_split_scale, with the chart point it stands
/// at and that point's lattice image.
template <typename Index, typename Int> struct hole_chart_rim_split {
  Index edge;
  std::uint8_t along;
  tf::point<double, 2> uv;
  tf::point<Int, 2> lattice;
};

/// The scratch one chart's sampling walks on, the refiner included, so a
/// worker reuses one triangulation carrier across the holes it takes.
/// `positions` holds, per refiner point, its position in
/// `[rim | samples]`; `found` the refiner points that split a rim edge.
template <typename Index, typename Int> struct hole_chart_sample_scratch {
  tf::blocked_buffer<Index, 2> constraints;
  tf::cdt_refiner<Index, Int> refiner;
  tf::buffer<Index> positions;
  tf::buffer<std::array<Index, 3>> found;
};

/// Sample the chart inside its rim and triangulate it, in one refinement.
///
/// The rim's lattice points are the refiner's input and its edges are its
/// constraints, so every sample is a Steiner point strictly inside the rim.
/// Where `split_rim` allows it the refinement also halves a rim edge its
/// quality asks to, and the rim it states is the extended one: each rim
/// position followed by the splits of its outgoing edge, in order along it.
/// The refiner splits a constraint at most @ref tf::hole_split_depth deep,
/// so every split is a parameter the fill's split records state exactly.
///
/// `triangles` are over `[extended rim | samples]` positions, each wound
/// positively in the chart; `splits` the extended rim's split positions in
/// its order; `samples` the chart points their lattice points read back, in
/// the order a triangle first names them. Returns false where two rim
/// positions share a lattice point or the rim does not triangulate.
template <typename Index, typename Int>
auto sample_hole_chart(
    const tf::fill::hole_collar<Index> &collar,
    const tf::fill::hole_chart<Int> &chart, bool split_rim,
    tf::fill::hole_chart_sample_scratch<Index, Int> &scratch,
    tf::buffer<tf::point<double, 2>> &samples,
    tf::buffer<std::array<Index, 3>> &triangles,
    tf::buffer<tf::fill::hole_chart_rim_split<Index, Int>> &splits) -> bool {
  using refiner_t = tf::cdt_refiner<Index, Int>;
  using param_t = typename refiner_t::param_t;
  static_assert(tf::topology::cdt::constrained_delaunay_refinement_owner<
                    Index, Int, Int>::max_split_depth == tf::hole_split_depth,
                "a rim split must be a parameter a split record states");
  const Index n = Index(collar.n_rim());
  samples.clear();
  triangles.clear();
  splits.clear();

  scratch.constraints.data_buffer().allocate(std::size_t(n) * 2);
  for (Index k = 0; k < n; ++k) {
    scratch.constraints[std::size_t(k)][0] = k;
    scratch.constraints[std::size_t(k)][1] = Index(k + 1 == n ? 0 : k + 1);
  }
  auto &refiner = scratch.refiner;
  if (split_rim)
    refiner.always_track_constraint_owners();
  if (!refiner.build(
          tf::make_points(tf::take(chart.lattice, std::size_t(n))),
          tf::make_edges(scratch.constraints),
          tf::cdt_refine_config{tf::fill::hole_chart_sample_quality, split_rim},
          tf::cdt_region_mode::nesting))
    return false;

  const auto lattice = refiner.points();
  scratch.positions.allocate(lattice.size());
  std::fill(scratch.positions.begin(), scratch.positions.end(), Index(-1));
  for (Index k = 0; k < n; ++k) {
    const Index out = refiner.index_map().f()[std::size_t(k)];
    if (out < 0 || std::size_t(out) >= lattice.size() ||
        scratch.positions[std::size_t(out)] >= 0)
      return false;
    scratch.positions[std::size_t(out)] = k;
  }

  const auto labels = refiner.region_labels();
  const Index n_faces = refiner.n_faces();
  Index extended = n;
  if (split_rim && refiner.n_constraint_splits()) {
    constexpr int bits = refiner_t::k_crossing_param_bits;
    const param_t whole = param_t(1) << bits;
    scratch.found.clear();
    for (Index f = 0; f < n_faces; ++f) {
      if (labels[std::size_t(f)] % 2 != 1)
        continue;
      const auto corners = refiner.face(f);
      const auto owners = refiner.face_constraint_owners(f);
      for (std::size_t e = 0; e < 3; ++e) {
        if (owners[e].input_id < 0)
          continue;
        const param_t ends[2] = {owners[e].t0, owners[e].t1};
        const Index at[2] = {corners[e], corners[(e + 1) % 3]};
        for (std::size_t side = 0; side < 2; ++side)
          if (ends[side] != param_t(0) && ends[side] != whole) {
            const param_t along = ends[side] >> (bits - tf::hole_split_depth);
            if ((along << (bits - tf::hole_split_depth)) != ends[side])
              return false;
            scratch.found.push_back(
                {Index(owners[e].input_id), Index(along), at[side]});
          }
      }
    }
    tbb::parallel_sort(scratch.found.begin(), scratch.found.end());
    scratch.found.erase_till_end(
        std::unique(scratch.found.begin(), scratch.found.end()));
    extended = Index(n + Index(scratch.found.size()));
    std::size_t next = 0;
    Index position = 0;
    for (Index k = 0; k < n; ++k) {
      scratch.positions[std::size_t(refiner.index_map().f()[std::size_t(k)])] =
          position++;
      for (; next < scratch.found.size() && scratch.found[next][0] == k;
           ++next) {
        const auto &split = scratch.found[next];
        if (scratch.positions[std::size_t(split[2])] >= 0)
          return false;
        scratch.positions[std::size_t(split[2])] = position++;
        const auto at = lattice[std::size_t(split[2])];
        const tf::point<Int, 2> point{at[0], at[1]};
        splits.push_back({k, std::uint8_t(split[1]),
                          chart.converter.deconvert(point), point});
      }
    }
  }

  for (Index f = 0; f < n_faces; ++f) {
    if (labels[std::size_t(f)] % 2 != 1)
      continue;
    const auto corners = refiner.face(f);
    std::array<Index, 3> named;
    for (std::size_t c = 0; c < 3; ++c) {
      auto &position = scratch.positions[std::size_t(corners[c])];
      if (position < 0) {
        position = Index(std::size_t(extended) + samples.size());
        const auto at = lattice[std::size_t(corners[c])];
        samples.push_back(
            chart.converter.deconvert(tf::point<Int, 2>{at[0], at[1]}));
      }
      named[c] = position;
    }
    triangles.push_back(named);
  }
  return true;
}

/// How the sampling refinement left the chart it last sampled.
template <typename Index, typename Int>
auto hole_chart_refine_status(
    const tf::fill::hole_chart_sample_scratch<Index, Int> &scratch)
    -> tf::hole_refine_status {
  switch (scratch.refiner.refine_status()) {
  case tf::cdt_refine_status::floor_met:
    return tf::hole_refine_status::floor_met;
  case tf::cdt_refine_status::stalled:
    return tf::hole_refine_status::stalled;
  case tf::cdt_refine_status::budget_exhausted:
    return tf::hole_refine_status::budget_exhausted;
  }
  return tf::hole_refine_status::stalled;
}

} // namespace tf::fill
