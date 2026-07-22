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
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/curves_buffer.hpp"
#include "../core/frame_of.hpp"
#include "../core/transformed.hpp"
#include "../cut/dispatch/arrangement.hpp"
#include "../cut/dispatch/boolean.hpp"
#include "../core/algorithm/parallel_transform.hpp"
#include "../core/buffer.hpp"
#include "../cut/construct/extract_intersection_curves.hpp"
#include "../exact/resolve_int_type.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./exact/make_kernel.hpp"
#include "./graph/intersection_graph.hpp"
#include "./intersect_config.hpp"
#include "./intersections_between_polygons.hpp"

namespace tf {

/// @ingroup intersect_curves
/// @brief Extract intersection curves where two meshes intersect.
///
/// Computes the geometric intersection between two polygon meshes and
/// returns the result as connected curves.
///
/// @tparam Policy0 The policy type for the first mesh.
/// @tparam Policy1 The policy type for the second mesh.
/// @param _polygons0 The first mesh @ref tf::polygons (or tagged form).
/// @param _polygons1 The second mesh @ref tf::polygons (or tagged form).
/// @param mode The intersection mode (sos or primitives).
/// @return A @ref tf::curves_buffer containing connected intersection curves.
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1>
auto make_intersection_curves(
    const tf::polygons<Policy0> &_polygons0,
    const tf::polygons<Policy1> &_polygons1,
    tf::intersect_config config = {tf::intersect_mode::sos}) {
  return cut::dispatch::boolean(
      _polygons0, _polygons1, [config](const auto &p0, const auto &p1) {
        using Index =
            std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                               typename std::decay_t<decltype(p1)>::index_type>;
        using InputReal = tf::coordinate_type<std::decay_t<decltype(p0)>,
                                              std::decay_t<decltype(p1)>>;
        using ResolvedInt = tf::exact::resolve_int_type<Int, InputReal>;
        using PipelineReal =
            std::conditional_t<std::is_integral_v<InputReal>, InputReal, double>;
        using RealOut =
            std::conditional_t<std::is_same_v<OutputCoordinateType, tf::none_t>,
                               InputReal, OutputCoordinateType>;
        static_assert(std::is_floating_point_v<RealOut> ||
                          std::is_integral_v<RealOut>,
                      "Output coordinate type must be floating-point or integral");
        static_assert(!std::is_integral_v<InputReal> ||
                          !std::is_floating_point_v<RealOut>,
                      "Integer input cannot produce floating-point output");

        tf::intersections_between_polygons<Index, PipelineReal, ResolvedInt> ibp;
        ibp.build(p0, p1, config);

        auto &conv = ibp.converter();
        auto apply_to_face = [&](int tag, Index object, const auto &f) {
          if (tag == 0)
            f(p0.faces()[object]);
          else
            f(p1.faces()[object]);
        };
        auto get_mesh_point = [&](int tag, Index id) -> tf::point<ResolvedInt, 3> {
          if (tag == 0)
            return conv.convert(
                tf::transformed(p0.points()[id], tf::frame_of(p0)));
          else
            return conv.convert(
                tf::transformed(p1.points()[id], tf::frame_of(p1)));
        };

        tf::intersection_graph<Index, ResolvedInt> ig;
        ig.build(ibp, apply_to_face, get_mesh_point, config.mode,
                 tf::exact::make_kernel(conv, config.tolerance));

        tf::curves_buffer<Index, RealOut, 3> cb;
        if (config.mode & tf::intersect_mode::sos) {
    tf::buffer<std::array<Index, 2>> sos_edges;
    sos_edges.allocate(std::size_t(ig.edge_groups().size()));
    tf::parallel_transform(
        ig.edge_groups(), tf::make_range(sos_edges),
        [](const auto &group) -> std::array<Index, 2> {
          return {group[0].point_0, group[0].point_1};
        });
    cb = tf::cut::curves_from_seam_edges<RealOut, Index>(sos_edges,
                                                         ig.points(), conv);
  } else {
          // regions, their coplanar collapse, and the seam scan — the
          // same read the arrangement paths use, no cut structure
          tf::buffer<Index> point_counts;
          point_counts.allocate(2);
          point_counts[0] = static_cast<Index>(p0.points().size());
          point_counts[1] = static_cast<Index>(p1.points().size());
          cb = tf::cut::make_region_curves<RealOut, Index>(
              ig, apply_to_face, get_mesh_point,
              tf::make_range(point_counts), conv);
        }
        if constexpr (!std::is_integral_v<InputReal> &&
                      std::is_integral_v<RealOut>) {
          auto conv_copy = ibp.converter();
          return std::make_tuple(std::move(cb), std::move(conv_copy));
        } else {
          return cb;
        }
      });
}

namespace intersect {

template <typename Int, typename OutputCoordinateType, typename FormsRange>
auto intersection_curves_n(const FormsRange &forms, tf::intersect_config config) {
  using Index = std::decay_t<decltype(forms[0].faces()[0][0])>;
  using InputReal = tf::coordinate_type<decltype(forms[0])>;
  using ResolvedInt = tf::exact::resolve_int_type<Int, InputReal>;
  using PipelineReal =
      std::conditional_t<std::is_integral_v<InputReal>, InputReal, double>;
  using RealOut =
      std::conditional_t<std::is_same_v<OutputCoordinateType, tf::none_t>,
                         InputReal, OutputCoordinateType>;
  static_assert(std::is_floating_point_v<RealOut> ||
                    std::is_integral_v<RealOut>,
                "Output coordinate type must be floating-point or integral");
  static_assert(!std::is_integral_v<InputReal> ||
                    !std::is_floating_point_v<RealOut>,
                "Integer input cannot produce floating-point output");

  tf::intersections_between_polygons<Index, PipelineReal, ResolvedInt> ibp;
  ibp.build(forms, config);

  auto &conv = ibp.converter();
  auto apply_to_face = [&](int tag, Index object, const auto &f) {
    f(forms[tag].faces()[object]);
  };
  auto get_mesh_point = [&](int tag, Index id) -> tf::point<ResolvedInt, 3> {
    return conv.convert(
        tf::transformed(forms[tag].points()[id], tf::frame_of(forms[tag])));
  };

  tf::intersection_graph<Index, ResolvedInt> ig;
  ig.build(ibp, apply_to_face, get_mesh_point, config.mode,
           tf::exact::make_kernel(conv, config.tolerance));

  tf::curves_buffer<Index, RealOut, 3> cb;
  if (config.mode & tf::intersect_mode::sos) {
    tf::buffer<std::array<Index, 2>> sos_edges;
    sos_edges.allocate(std::size_t(ig.edge_groups().size()));
    tf::parallel_transform(
        ig.edge_groups(), tf::make_range(sos_edges),
        [](const auto &group) -> std::array<Index, 2> {
          return {group[0].point_0, group[0].point_1};
        });
    cb = tf::cut::curves_from_seam_edges<RealOut, Index>(sos_edges,
                                                         ig.points(), conv);
  } else {
    // regions, their coplanar collapse, and the seam scan — the same
    // read the arrangement paths use, no cut structure
    tf::buffer<Index> point_counts;
    point_counts.allocate(std::size_t(forms.size()));
    for (std::size_t t = 0; t < std::size_t(forms.size()); ++t)
      point_counts[t] = static_cast<Index>(forms[t].points().size());
    cb = tf::cut::make_region_curves<RealOut, Index>(
        ig, apply_to_face, get_mesh_point, tf::make_range(point_counts), conv);
  }

  if constexpr (!std::is_integral_v<InputReal> &&
                std::is_integral_v<RealOut>) {
    auto conv_copy = ibp.converter();
    return std::make_tuple(std::move(cb), std::move(conv_copy));
  } else {
    return cb;
  }
}

} // namespace intersect

/// @ingroup intersect_curves
/// @brief Extract intersection curves from N meshes.
///
/// Computes all pairwise intersections among the given meshes and
/// returns the result as connected curves.
///
/// @tparam Range A range of @ref tf::polygons (or tagged forms).
/// @param _forms The input meshes.
/// @param mode The intersection mode (sos or primitives).
/// @return A @ref tf::curves_buffer containing connected intersection curves.
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Range>
auto make_intersection_curves(
    const Range &_forms,
    tf::intersect_config config = {tf::intersect_mode::sos |
                                   tf::intersect_mode::resolve_crossing_contours}) {
  return cut::dispatch::arrangement(_forms, [config](const auto &forms) {
    return intersect::intersection_curves_n<Int, OutputCoordinateType>(forms,
                                                                       config);
  });
}

} // namespace tf
