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
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/coordinate_dims.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/curves_buffer.hpp"
#include "../core/edges.hpp"
#include "../core/none.hpp"
#include "../core/polygons.hpp"
#include "../core/range.hpp"
#include "../core/return_refused.hpp"
#include "../intersect/make_intersection_edges.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./cut/build_iso_cuts.hpp"
#include "./cut/embedded_isocurves.hpp"

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tf {

namespace iso {

/// Shared worker for the embedding entry points: the cut is one pipeline and
/// the overloads differ in nothing but what they keep. `IndexRequest` is
/// `tf::none_t` (deduce from the faces) or the caller's index type; `Refused`
/// and `Curves` are each `tf::none_t` (skip) or their request tag, and the
/// returned tuple grows accordingly.
template <typename IndexRequest, typename Refused, typename Curves,
          typename Policy, typename Range0, typename Iterator0, std::size_t N0>
auto isocurves_worker(const tf::polygons<Policy> &polygons,
                      const Range0 &scalars,
                      const tf::range<Iterator0, N0> &cut_values) {
  using Index =
      std::conditional_t<std::is_same_v<IndexRequest, tf::none_t>,
                         std::decay_t<decltype(polygons.faces()[0][0])>,
                         IndexRequest>;
  auto [sfi, regions, pids] =
      tf::iso::build_iso_cuts<Index>(polygons, scalars, cut_values);
  auto [res_polygons, labels, face_labels] =
      tf::iso::embedded_isocurves<Index>(polygons, sfi, regions, pids);

  if constexpr (!std::is_same_v<Refused, tf::none_t>) {
    return std::make_tuple(std::move(res_polygons), std::move(labels),
                           std::move(face_labels), std::move(regions.refused));
  } else if constexpr (!std::is_same_v<Curves, tf::none_t>) {
    auto ie = tf::make_intersection_edges(sfi, polygons.faces());
    tf::curves_buffer<Index, tf::coordinate_type<Policy>,
                      tf::coordinate_dims_v<Policy>>
        cb;
    cb.paths_buffer() = tf::connect_edges_to_paths(tf::make_edges(ie));
    cb.points_buffer().allocate(sfi.intersection_points().size());
    tf::parallel_copy(sfi.intersection_points(), cb.points());
    return std::make_tuple(std::move(res_polygons), std::move(labels),
                           std::move(face_labels), std::move(cb));
  } else {
    return std::make_tuple(std::move(res_polygons), std::move(labels),
                           std::move(face_labels));
  }
}

} // namespace iso

/// @ingroup iso
/// @brief Embed scalar field isocurves into mesh topology.
///
/// Creates a new mesh where isocurve contour lines become edges.
/// Faces are split along the curves and labeled by isoband region.
/// Labels indicate which region between consecutive cut values each face
/// belongs to (0 for below first cut, 1 for between first and second, etc.).
///
/// @tparam Index The index type (auto-deduced if not specified).
/// @tparam Policy The policy type of the polygons.
/// @param polygons The input @ref tf::polygons.
/// @param scalars The scalar field values (one per vertex).
/// @param cut_values The threshold values to embed.
/// @return Tuple of (@ref tf::polygons_buffer, labels buffer).
///
/// @see tf::make_isobands for extracting specific bands only.
template <typename Index = tf::none_t, typename Policy, typename Range0,
          typename Iterator0, std::size_t N0>
auto embedded_isocurves(const tf::polygons<Policy> &polygons,
                        const Range0 &scalars,
                        const tf::range<Iterator0, N0> &cut_values) {
  return iso::isocurves_worker<Index, tf::none_t, tf::none_t>(polygons, scalars,
                                                              cut_values);
}

/// @ingroup iso
/// @brief Embed scalar field isocurves and name the cut faces the
///        triangulation refused.
///
/// The mesh, labels and face labels are exactly the ones the untagged call
/// returns. A refused face holds no piece, so its surface is absent from the
/// result — emptiness is the answer, and this overload names whose emptiness it
/// was. The one-chord split cannot refuse (it declines to the constrained build
/// instead), and a constraint set the field states on a face it crosses is
/// recoverable, so the list speaks only on degenerate input.
///
/// @return Tuple of (@ref tf::polygons_buffer, labels, face labels, ascending
///         refused face ids).
template <typename Index = tf::none_t, typename Policy, typename Range0,
          typename Iterator0, std::size_t N0>
auto embedded_isocurves(const tf::polygons<Policy> &polygons,
                        const Range0 &scalars,
                        const tf::range<Iterator0, N0> &cut_values,
                        tf::return_refused_t) {
  return iso::isocurves_worker<Index, tf::return_refused_t, tf::none_t>(
      polygons, scalars, cut_values);
}

/// @ingroup iso
/// @brief Embed scalar field isocurves into mesh topology with curve output.
/// @overload
template <typename Index = tf::none_t, typename Policy, typename Range0,
          typename Iterator0, std::size_t N0>
auto embedded_isocurves(const tf::polygons<Policy> &polygons,
                        const Range0 &scalars,
                        const tf::range<Iterator0, N0> &cut_values,
                        tf::return_curves_t) {
  return iso::isocurves_worker<Index, tf::none_t, tf::return_curves_t>(
      polygons, scalars, cut_values);
}

} // namespace tf
