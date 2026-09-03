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

#include "../core/coordinate_type.hpp"
#include "../core/edges.hpp"
#include "../core/none.hpp"
#include "../core/points.hpp"
#include "../core/range.hpp"
#include "../exact/resolve_int_type.hpp"
#include "../reindex/return_index_map.hpp"
#include "./cdt/make_constrained_delaunay.hpp"
#include "./cdt/make_unconstrained_delaunay.hpp"
#include "./cdt_config.hpp"
#include "./return_region_labels.hpp"
#include <cstddef>
#include <type_traits>

namespace tf {

/// @ingroup topology
/// @brief Constrained Delaunay triangulation as a one-shot helper.
///
/// Returns a 2D `polygons_buffer` of the exact-welded input points, augmented
/// with intersection vertices when constraint edges cross. With constraint
/// edges, only triangles inside the
/// constrained outlines are returned (parity = 1 across each crossing
/// of a constrained edge); without edges, the full convex hull
/// triangulation is returned.
///
/// `config.split_constraints = true` resolves a crossing where the constraint
/// walk meets it, adding an output point there. `false` preserves the
/// constraints verbatim and returns an empty result if any two of them cross —
/// that emptiness is the answer to "do these constraints cross", not an
/// error. A constraint whose endpoints coincide carries no length and is
/// dropped either way. `config` is implicitly constructible from `bool`
/// alone, so a call site passing only `split_constraints` still reads as
/// before; `config.regions` matters only to the
/// @ref tf::return_region_labels overloads.
///
/// @tparam Index The index type, deduced from `edges[0][0]` when constraint
///   edges are given. The overloads that take no edges have nothing to deduce
///   it from and default to `int`.
/// @tparam Int The exact-arithmetic integer type. Pass `tf::none_t`
///   (default) to auto-resolve to int32 for float input or int64 for
///   double input.
/// @tparam PointsPolicy Deduced from input points.
/// @tparam EdgesPolicy Deduced from input edges.
template <typename Index = tf::none_t, typename Int = tf::none_t,
          typename PointsPolicy, typename EdgesPolicy>
auto make_cdt(const tf::points<PointsPolicy> &pts,
              const tf::edges<EdgesPolicy> &edges, cdt_config config = {}) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(edges[0][0])>;
    return make_cdt<ActualIndex, Int>(pts, edges, config);
  } else {
    using coord_t = tf::coordinate_type<PointsPolicy>;
    using ResolvedInt = tf::exact::resolve_int_type<Int, coord_t>;
    return tf::topology::cdt::make_constrained_delaunay<false, false, Index,
                                                        coord_t, ResolvedInt>(
        pts, edges, config);
  }
}

/// @overload
/// @brief Convex-hull triangulation with no constraint edges.
template <typename Index = int, typename Int = tf::none_t,
          typename PointsPolicy>
auto make_cdt(const tf::points<PointsPolicy> &pts) {
  using coord_t = tf::coordinate_type<PointsPolicy>;
  using ResolvedInt = tf::exact::resolve_int_type<Int, coord_t>;
  return tf::topology::cdt::make_unconstrained_delaunay<false, Index, coord_t,
                                                        ResolvedInt>(pts);
}

/// @overload
/// @brief As above; also returns the input-to-output index map.
template <typename Index = tf::none_t, typename Int = tf::none_t,
          typename PointsPolicy, typename EdgesPolicy>
auto make_cdt(const tf::points<PointsPolicy> &pts,
              const tf::edges<EdgesPolicy> &edges, tf::return_index_map_t,
              cdt_config config = {}) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(edges[0][0])>;
    return make_cdt<ActualIndex, Int>(pts, edges, tf::return_index_map, config);
  } else {
    using coord_t = tf::coordinate_type<PointsPolicy>;
    using ResolvedInt = tf::exact::resolve_int_type<Int, coord_t>;
    return tf::topology::cdt::make_constrained_delaunay<true, false, Index,
                                                        coord_t, ResolvedInt>(
        pts, edges, config);
  }
}

/// @overload
/// @brief The unfiltered triangulation plus its per-triangle region labels.
///
/// No interior filter is applied: every triangle of the triangulation is
/// returned, and the labels state its region in `config.regions` mode —
/// nesting parity or wall-cut component id (0 the hull exterior). The
/// default entries are exactly this read filtered by nesting parity.
template <typename Index = tf::none_t, typename Int = tf::none_t,
          typename PointsPolicy, typename EdgesPolicy>
auto make_cdt(const tf::points<PointsPolicy> &pts,
              const tf::edges<EdgesPolicy> &edges, tf::return_region_labels_t,
              cdt_config config = {}) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(edges[0][0])>;
    return make_cdt<ActualIndex, Int>(pts, edges, tf::return_region_labels,
                                      config);
  } else {
    using coord_t = tf::coordinate_type<PointsPolicy>;
    using ResolvedInt = tf::exact::resolve_int_type<Int, coord_t>;
    return tf::topology::cdt::make_constrained_delaunay<false, true, Index,
                                                        coord_t, ResolvedInt>(
        pts, edges, config);
  }
}

/// @overload
/// @brief As above; also returns the input-to-output index map.
template <typename Index = tf::none_t, typename Int = tf::none_t,
          typename PointsPolicy, typename EdgesPolicy>
auto make_cdt(const tf::points<PointsPolicy> &pts,
              const tf::edges<EdgesPolicy> &edges, tf::return_region_labels_t,
              tf::return_index_map_t, cdt_config config = {}) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(edges[0][0])>;
    return make_cdt<ActualIndex, Int>(pts, edges, tf::return_region_labels,
                                      tf::return_index_map, config);
  } else {
    using coord_t = tf::coordinate_type<PointsPolicy>;
    using ResolvedInt = tf::exact::resolve_int_type<Int, coord_t>;
    return tf::topology::cdt::make_constrained_delaunay<true, true, Index,
                                                        coord_t, ResolvedInt>(
        pts, edges, config);
  }
}

/// @overload
/// @brief Constrained Delaunay triangulation with an explicit per-edge
///   region-boundary mask.
///
/// `is_boundary[i] == true` marks the constraint as a region wall —
/// `region_labels()` parity flips when crossed, so it participates in
/// inside/outside classification. `false` marks the constraint as
/// preserved-but-not-boundary (held in the triangulation but not part
/// of any region wall).
template <typename Index = tf::none_t, typename Int = tf::none_t,
          typename PointsPolicy, typename EdgesPolicy, typename Iterator,
          std::size_t N>
auto make_cdt(const tf::points<PointsPolicy> &pts,
              const tf::edges<EdgesPolicy> &edges,
              const tf::range<Iterator, N> &is_boundary,
              cdt_config config = {}) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(edges[0][0])>;
    return make_cdt<ActualIndex, Int>(pts, edges, is_boundary, config);
  } else {
    using coord_t = tf::coordinate_type<PointsPolicy>;
    using ResolvedInt = tf::exact::resolve_int_type<Int, coord_t>;
    return tf::topology::cdt::make_constrained_delaunay<false, false, Index,
                                                        coord_t, ResolvedInt>(
        pts, edges, config, is_boundary);
  }
}

/// @overload
/// @brief As above; also returns the input-to-output index map.
template <typename Index = tf::none_t, typename Int = tf::none_t,
          typename PointsPolicy, typename EdgesPolicy, typename Iterator,
          std::size_t N>
auto make_cdt(const tf::points<PointsPolicy> &pts,
              const tf::edges<EdgesPolicy> &edges,
              const tf::range<Iterator, N> &is_boundary, tf::return_index_map_t,
              cdt_config config = {}) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(edges[0][0])>;
    return make_cdt<ActualIndex, Int>(pts, edges, is_boundary,
                                      tf::return_index_map, config);
  } else {
    using coord_t = tf::coordinate_type<PointsPolicy>;
    using ResolvedInt = tf::exact::resolve_int_type<Int, coord_t>;
    return tf::topology::cdt::make_constrained_delaunay<true, false, Index,
                                                        coord_t, ResolvedInt>(
        pts, edges, config, is_boundary);
  }
}

/// @overload
/// @brief Masked walls; the unfiltered triangulation plus its labels.
template <typename Index = tf::none_t, typename Int = tf::none_t,
          typename PointsPolicy, typename EdgesPolicy, typename Iterator,
          std::size_t N>
auto make_cdt(const tf::points<PointsPolicy> &pts,
              const tf::edges<EdgesPolicy> &edges,
              const tf::range<Iterator, N> &is_boundary,
              tf::return_region_labels_t, cdt_config config = {}) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(edges[0][0])>;
    return make_cdt<ActualIndex, Int>(pts, edges, is_boundary,
                                      tf::return_region_labels, config);
  } else {
    using coord_t = tf::coordinate_type<PointsPolicy>;
    using ResolvedInt = tf::exact::resolve_int_type<Int, coord_t>;
    return tf::topology::cdt::make_constrained_delaunay<false, true, Index,
                                                        coord_t, ResolvedInt>(
        pts, edges, config, is_boundary);
  }
}

/// @overload
/// @brief As above; also returns the input-to-output index map.
template <typename Index = tf::none_t, typename Int = tf::none_t,
          typename PointsPolicy, typename EdgesPolicy, typename Iterator,
          std::size_t N>
auto make_cdt(const tf::points<PointsPolicy> &pts,
              const tf::edges<EdgesPolicy> &edges,
              const tf::range<Iterator, N> &is_boundary,
              tf::return_region_labels_t, tf::return_index_map_t,
              cdt_config config = {}) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(edges[0][0])>;
    return make_cdt<ActualIndex, Int>(pts, edges, is_boundary,
                                      tf::return_region_labels,
                                      tf::return_index_map, config);
  } else {
    using coord_t = tf::coordinate_type<PointsPolicy>;
    using ResolvedInt = tf::exact::resolve_int_type<Int, coord_t>;
    return tf::topology::cdt::make_constrained_delaunay<true, true, Index,
                                                        coord_t, ResolvedInt>(
        pts, edges, config, is_boundary);
  }
}

/// @overload
/// @brief Convex-hull triangulation with no edges; returns the index map.
template <typename Index = int, typename Int = tf::none_t,
          typename PointsPolicy>
auto make_cdt(const tf::points<PointsPolicy> &pts, tf::return_index_map_t) {
  using coord_t = tf::coordinate_type<PointsPolicy>;
  using ResolvedInt = tf::exact::resolve_int_type<Int, coord_t>;
  return tf::topology::cdt::make_unconstrained_delaunay<true, Index, coord_t,
                                                        ResolvedInt>(pts);
}

} // namespace tf
