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
#include "../arrangement/return_curves.hpp"
#include "../core/coordinate_dims.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/curves_buffer.hpp"
#include "../core/edges.hpp"
#include "../core/none.hpp"
#include "../core/polygons.hpp"
#include "../core/range.hpp"
#include "../core/return_refused.hpp"
#include "../core/segments.hpp"
#include "../intersect/make_intersection_edges.hpp"
#include "../reindex/by_ids_on_points.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./cut/build_iso_cuts.hpp"
#include "./cut/make_isobands.hpp"

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tf {

namespace iso {

/// Shared worker for the band entry points: the cut is one pipeline and the
/// overloads differ in nothing but what they keep. `IndexRequest` is
/// `tf::none_t` (deduce from the faces) or the caller's index type; `Refused`
/// and `Curves` are each `tf::none_t` (skip) or their request tag, and the
/// returned tuple grows accordingly. The curves are the seams of the KEPT
/// bands alone, so they are filtered by the points the emitter created.
template <typename IndexRequest, typename Refused, typename Curves,
          typename Policy, typename Range0, typename Iterator0, std::size_t N0,
          typename Iterator1, std::size_t N1>
auto isobands_worker(const tf::polygons<Policy> &polygons,
                     const Range0 &scalars,
                     const tf::range<Iterator0, N0> &cut_values,
                     const tf::range<Iterator1, N1> &selected_bands) {
  using Index =
      std::conditional_t<std::is_same_v<IndexRequest, tf::none_t>,
                         std::decay_t<decltype(polygons.faces()[0][0])>,
                         IndexRequest>;
  auto [sfi, regions, pids] =
      tf::iso::build_iso_cuts<Index>(polygons, scalars, cut_values);
  auto [res_polygons, labels, face_labels, created_ids] =
      tf::iso::make_isobands<Index>(polygons, sfi, regions, pids,
                                    selected_bands);

  if constexpr (!std::is_same_v<Refused, tf::none_t>) {
    return std::make_tuple(std::move(res_polygons), std::move(labels),
                           std::move(face_labels), std::move(regions.refused));
  } else if constexpr (!std::is_same_v<Curves, tf::none_t>) {
    auto ie = tf::make_intersection_edges(sfi, polygons.faces());
    auto all_segments =
        tf::make_segments(tf::make_edges(ie), sfi.intersection_points());
    auto filtered_segments =
        tf::reindexed_by_ids_on_points(all_segments, created_ids);

    tf::curves_buffer<Index, tf::coordinate_type<Policy>,
                      tf::coordinate_dims_v<Policy>>
        cb;
    cb.paths_buffer() = tf::connect_edges_to_paths(filtered_segments.edges());
    cb.points_buffer() = std::move(filtered_segments.points_buffer());
    return std::make_tuple(std::move(res_polygons), std::move(labels),
                           std::move(face_labels), std::move(cb));
  } else {
    return std::make_tuple(std::move(res_polygons), std::move(labels),
                           std::move(face_labels));
  }
}

} // namespace iso

/// @ingroup iso
/// @brief Extract specific isobands from a scalar field.
///
/// Like @ref tf::embedded_isocurves but returns only the selected
/// bands between specified cut values. Useful for extracting regions
/// within specific value ranges.
///
/// @tparam Index The index type (auto-deduced if not specified).
/// @tparam Policy The policy type of the polygons.
/// @param polygons The input @ref tf::polygons.
/// @param scalars The scalar field values (one per vertex).
/// @param cut_values The threshold values defining band boundaries.
/// @param selected_bands Indices of bands to extract.
/// @return Tuple of (@ref tf::polygons_buffer, labels buffer).
///
/// @see tf::embedded_isocurves for embedding all bands.
template <typename Index = tf::none_t, typename Policy, typename Range0,
          typename Iterator0, std::size_t N0, typename Iterator1,
          std::size_t N1>
auto make_isobands(const tf::polygons<Policy> &polygons, const Range0 &scalars,
                   const tf::range<Iterator0, N0> &cut_values,
                   const tf::range<Iterator1, N1> &selected_bands) {
  return iso::isobands_worker<Index, tf::none_t, tf::none_t>(
      polygons, scalars, cut_values, selected_bands);
}

/// @ingroup iso
/// @brief Extract specific isobands and name the cut faces the triangulation
///        refused.
///
/// The mesh, labels and face labels are exactly the ones the untagged call
/// returns, and the refused ids are the whole mesh's — a face the cut declined
/// holds no piece in any band. Emptiness is the answer; this overload names
/// whose emptiness it was. The one-chord split cannot refuse (it declines to
/// the constrained build instead), and a constraint set the field states on a
/// face it crosses is recoverable, so the list speaks only on degenerate input.
///
/// @return Tuple of (@ref tf::polygons_buffer, labels, face labels, ascending
///         refused face ids).
template <typename Index = tf::none_t, typename Policy, typename Range0,
          typename Iterator0, std::size_t N0, typename Iterator1,
          std::size_t N1>
auto make_isobands(const tf::polygons<Policy> &polygons, const Range0 &scalars,
                   const tf::range<Iterator0, N0> &cut_values,
                   const tf::range<Iterator1, N1> &selected_bands,
                   tf::return_refused_t) {
  return iso::isobands_worker<Index, tf::return_refused_t, tf::none_t>(
      polygons, scalars, cut_values, selected_bands);
}

/// @ingroup iso
/// @brief Extract specific isobands from a scalar field with curve output.
/// @overload
template <typename Index = tf::none_t, typename Policy, typename Range0,
          typename Iterator0, std::size_t N0, typename Iterator1,
          std::size_t N1>
auto make_isobands(const tf::polygons<Policy> &polygons, const Range0 &scalars,
                   const tf::range<Iterator0, N0> &cut_values,
                   const tf::range<Iterator1, N1> &selected_bands,
                   tf::return_curves_t) {
  return iso::isobands_worker<Index, tf::none_t, tf::return_curves_t>(
      polygons, scalars, cut_values, selected_bands);
}

} // namespace tf
