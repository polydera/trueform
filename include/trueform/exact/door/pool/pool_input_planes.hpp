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

#include "./build_name_registry.hpp"
#include "./chart_cell.hpp"
#include "./exact_lane.hpp"
#include "./phase_clock.hpp"
#include "./pool_records.hpp"
#include "./wide_to_double.hpp"

#include "../../../core/algorithm/parallel_fill.hpp"
#include "../../../core/algorithm/parallel_for_each.hpp"
#include "../../../core/buffer.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/offset_block_buffer.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "../../canonical_plane.hpp"
#include "../../meta.hpp"
#include "../placement_tables.hpp"
#include "../round_to_wide.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <tbb/parallel_sort.h>

namespace tf::exact::door::pool {

/// THE PARTITION AS THE PLACEMENT READS IT: per flat face, the pool its exact
/// name committed to, or `-1`, and per pool the committed plane. A face that
/// pooled answers to that plane from then on — its corners and every vertex
/// admitted against it land on it — which is how a wall the partition made
/// flat stays flat.
template <typename Int> struct input_pools {
  tf::buffer<int> pool_of_face;
  tf::buffer<tf::exact::canonical_plane<Int>> plane;
  election_census election;
  certificate_census certificate;
};

/// One name's key and the name: the sort gathers a key's names into one run.
struct plane_key_record {
  chart_cell cell{};
  std::int64_t offset = 0;
  int name = -1;

  auto operator<(const plane_key_record &o) const -> bool {
    if (cell != o.cell)
      return cell < o.cell;
    if (offset != o.offset)
      return offset < o.offset;
    return name < o.name;
  }
  auto same_key(const plane_key_record &o) const -> bool {
    return cell == o.cell && offset == o.offset;
  }
};

/// THE POOLING IS A QUANTIZATION, SO IT IS TRANSITIVE. A name is an exact
/// plane, and its key is that plane rounded: the direction onto the fixed
/// chart, the intercept along the cell's own direction onto the band. Equal
/// keys are one pool, whatever the forms, and a key one name holds alone
/// pools nothing — no partner, no motion. Nothing walks from a name to a
/// neighbour, so a curved surface, whose faces round into ever different
/// cells, never welds across itself.
///
/// The chart's pitch is the scene's: at `K` steps a direction in a cell is
/// within `1 / 2K` of the cell's own, so over the lattice's whole extent
/// the planes of one cell part by at most a quarter band, and with the
/// intercept rounded at the band every member stands within three quarters
/// of it from the pool's plane. The pool's plane is the cell's integer
/// direction, dense on the lattice, through the cell's centre offset on the
/// nearest offset the lattice realizes.
template <typename Index, typename Int, typename RealType>
auto pool_input_planes(const placement_tables<Index, Int, RealType> &tables,
                       Int tolerance, input_pools<Int> &pools) -> void {
  using T2 = typename tf::exact::meta<Int>::T2;
  using coefficient_type = typename exact_lane<Int>::coefficient_type;
  phase_clock scene;
  pool_names<Int> names;
  tf::buffer<int> face_name;
  tf::offset_block_buffer<int, Index> name_faces;
  build_name_registry<Index, Int>(tables.planes, names.plane, face_name,
                                  name_faces);
  pools.election.scene_seconds = scene.since();
  pools.election.names = std::int64_t(names.plane.size());
  const auto n_names = names.plane.size();

  // the pitch: a quarter band over the lattice's reach from the origin
  const double reach =
      std::sqrt(3.0) * std::ldexp(1.0, std::numeric_limits<Int>::digits);
  const double band = double(tolerance);
  const double steps = std::ceil(4.0 * reach / band);
  const int resolution =
      steps < 4.0 ? 4
                  : (steps > double(std::numeric_limits<int>::max() / 4)
                         ? std::numeric_limits<int>::max() / 4
                         : int(steps));
  const auto integer_normal = [resolution](const chart_cell &cell) {
    std::array<std::int64_t, 3> n{};
    std::size_t held = 1;
    for (std::size_t k = 0; k < 3; ++k) {
      if (int(k) == cell.at[0])
        n[k] = resolution;
      else
        n[k] = cell.at[held++];
    }
    return n;
  };

  tf::buffer<plane_key_record> record;
  record.allocate(n_names);
  tf::parallel_for_each(
      tf::make_sequence_range(n_names),
      [&](std::size_t n) {
        const auto &plane = names.plane[n];
        const std::array<coefficient_type, 3> normal{
            coefficient_type(plane[0]), coefficient_type(plane[1]),
            coefficient_type(plane[2])};
        plane_key_record made;
        made.name = int(n);
        made.cell = chart_cell_of<Int>(normal, resolution);
        if (made.cell.sign == 0) {
          record[n] = made;
          return;
        }
        const auto axis = integer_normal(made.cell);
        double dot = 0.0, length2 = 0.0;
        for (std::size_t k = 0; k < 3; ++k) {
          dot += wide_to_double(plane[k]) * double(axis[k]);
          length2 += double(axis[k]) * double(axis[k]);
        }
        // the intercept of the plane along the cell's own ray, in lattice
        // units; the sign of the alignment cancels between offset and dot
        const double intercept =
            wide_to_double(plane[3]) * std::sqrt(length2) / dot;
        made.offset = std::int64_t(std::llround(intercept / band));
        record[n] = made;
      },
      tf::checked);
  tbb::parallel_sort(record.begin(), record.end());

  tf::buffer<int> pool_of_name;
  pool_of_name.allocate(n_names);
  tf::parallel_fill(pool_of_name, -1);
  pools.plane.clear();
  std::int64_t cells = 0;
  for (std::size_t lo = 0; lo < n_names;) {
    std::size_t hi = lo + 1;
    while (hi < n_names && record[hi].same_key(record[lo]))
      ++hi;
    if (record[lo].cell.sign != 0) {
      ++cells;
      if (hi - lo >= 2) {
        const auto axis = integer_normal(record[lo].cell);
        double length2 = 0.0;
        for (const auto a : axis)
          length2 += double(a) * double(a);
        const std::int64_t g = std::gcd(
            std::gcd(std::llabs(axis[0]), std::llabs(axis[1])),
            std::llabs(axis[2]));
        const double centre = double(record[lo].offset) * band;
        const double offset = std::round(centre * std::sqrt(length2) /
                                         double(g)) *
                              double(g);
        tf::exact::canonical_plane<Int> plane{
            T2(axis[0]), T2(axis[1]), T2(axis[2]),
            T2(tf::exact::door::round_to_wide<Int>(offset))};
        const int pool = int(pools.plane.size());
        pools.plane.push_back(tf::exact::canonicalize_plane<Int>(plane));
        for (std::size_t k = lo; k < hi; ++k)
          pool_of_name[std::size_t(record[k].name)] = pool;
      }
    }
    lo = hi;
  }
  pools.election.cells = cells;
  pools.election.pools = std::int64_t(pools.plane.size());
  pools.election.cell_seconds = scene.since();

  // one disjoint store per face, so the face is the grain
  pools.pool_of_face.allocate(face_name.size());
  tf::parallel_for_each(
      tf::make_sequence_range(face_name.size()),
      [&face_name, &pool_of_name, &pools](std::size_t f) {
        const int name = face_name[f];
        pools.pool_of_face[f] =
            name < 0 ? -1 : pool_of_name[std::size_t(name)];
      },
      tf::checked);
}

} // namespace tf::exact::door::pool
