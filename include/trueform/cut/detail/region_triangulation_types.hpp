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
#include <cstdint>

#include "../../core/array_hash.hpp"
#include "../../core/buffer.hpp"
#include "../../core/index_hash_map.hpp"
#include "../../core/point.hpp"
#include "../../core/points_buffer.hpp"
#include "../../exact/meta.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../../topology/constrained_delaunay_triangulator.hpp"
#include <array>

namespace tf::cut::detail {

inline constexpr int region_triangulation_max_rounds = 4;

/// What a call to the one triangulation gate is for.
///   assemble — fill the workspace only, no build.
///   preserve — build refusing on any constraint crossing; refusal is the
///              signal that enters a region into the recovery wave.
///   resolve  — build resolving crossings; the wave's last word.
enum class region_build : char { assemble, preserve, resolve };

/// Resolution of the double-precision ring the conforming pass builds for
/// local feature size. A split enters that ring at this width, and every
/// bucket count and round-to-nearest bias below follows from it.
inline constexpr int ring_bucket_bits = 8;
inline constexpr std::uint32_t ring_buckets = std::uint32_t(1)
                                              << ring_bucket_bits;

template <typename Index> struct region_vertex_hash {
  auto operator()(const tf::intersect::graph::vertex<Index> &vertex) const {
    return hash(
        std::array<Index, 2>{Index(vertex.source), vertex.id});
  }
  tf::array_hash<Index, 2> hash;
};

template <typename Index> struct region_triangulation_merge {
  std::array<Index, 2> from;
  std::array<Index, 2> to_key;
  tf::intersect::graph::vertex<Index> to;
};

template <typename Index, typename Int> struct region_split_proposal {
  std::array<tf::intersect::graph::vertex<Index>, 2> edge;
  Index tag;
  typename tf::exact::meta<Int>::param_type param;
};

template <typename Index> struct region_crossing_proposal {
  Index tag;
  Index n_parents;
  char t_junction;
  tf::intersect::graph::vertex<Index> v;
};

/// A split of a physical edge, transportable to every carrier of that
/// edge. `param` is a position along the edge, so its width follows the
/// lattice precision through `meta<Int>` — never the index type.
template <typename Index, typename Int> struct region_triangulation_split {
  std::array<std::array<Index, 2>, 2> edge;
  typename tf::exact::meta<Int>::param_type param;
  std::array<Index, 2> key;
  tf::intersect::graph::vertex<Index> vertex;
  char named = 0;
};

/// An interior vertex of a region together with its position. The position
/// travels with the vertex because a point refinement has just chosen has no
/// id yet, so nothing can look it up.
template <typename Index, typename Int> struct region_interior_point {
  tf::intersect::graph::vertex<Index> vertex;
  tf::point<Int, 3> position;
};

template <typename Index> struct promoted_face_candidates {
  tf::buffer<std::array<Index, 2>> faces;
  tf::buffer<std::array<Index, 2>> steiner_ranges;
};

template <typename Index, typename Int> struct region_refinement_discovery {
  tf::buffer<region_split_proposal<Index, Int>> proposals;
  tf::buffer<tf::point<Int, 3>> steiner_points;
  tf::buffer<std::array<Index, 2>> steiner_counts;
  Index refused = 0;
  Index recovered = 0;
};

template <typename Index> struct materialized_region_steiners {
  tf::buffer<Index> offsets;
  Index base = 0;
};

template <typename Index, typename Int> struct region_triangulation_workspace {
  tf::constrained_delaunay_triangulator<Index, Int, Int> tri;
  tf::index_hash_map<tf::intersect::graph::vertex<Index>, Index,
                     region_vertex_hash<Index>>
      ihm;
  tf::points_buffer<Int, 2> pts;
  tf::buffer<Index> cons;
  tf::buffer<std::array<tf::intersect::graph::vertex<Index>, 2>>
      cons_verts;
  tf::buffer<Index> cons_slot;
  tf::buffer<char> cons_sub;
  tf::buffer<std::array<tf::intersect::graph::vertex<Index>, 2>>
      cons_parent;
  /// Parameter interval of each constraint piece along its physical
  /// edge — a position, so it follows the lattice precision.
  tf::buffer<std::array<typename tf::exact::meta<Int>::param_type, 2>> cons_t;
  tf::buffer<std::array<Index, 3>> split_seen;
  tf::buffer<char> pos_set;
  tf::buffer<Index> rep;
  tf::buffer<region_triangulation_merge<Index>> merges;
  int ax0 = 0;
  int ax1 = 1;
  bool flip = false;
  std::array<typename tf::exact::meta<Int>::T2, 3> plane_n{};
  typename tf::exact::meta<Int>::T2 plane_d{};
  Int plane_fallback{};
  Index memo_tag = -1;
  Index memo_object = -1;
  Index face_size = 3;
};

} // namespace tf::cut::detail
